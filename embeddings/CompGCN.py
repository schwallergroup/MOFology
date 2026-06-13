import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rdflib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rdflib import Graph
from torch_geometric.nn import MessagePassing
from torch.utils.checkpoint import checkpoint as ckpt_fn


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_TTL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a real CompGCN model on MOF KG.")
    parser.add_argument("--ttl_path", type=str, default=DEFAULT_TTL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_bases", type=int, default=16, help="Number of bases for relation decomposition.")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Ratio of triples for validation (MRR).")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--decoder", type=str, choices=["distmult", "transe"], default="distmult")
    parser.add_argument("--comp_op", type=str, choices=["sub", "mult", "corr"], default="mult")
    parser.add_argument("--margin", type=float, default=1.0, help="Used for TransE-style margin loss.")
    parser.add_argument("--edge_dropout", type=float, default=0.3, help="Fraction of edges to DROP each epoch.")
    parser.add_argument("--num_sub_batches", type=int, default=4, help="Sub-batches of triples scored per GNN forward.")
    parser.add_argument("--checkpoint_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1,
        help="Epoch number to start/resume from (set to N+1 when resuming from epoch N).",
    )
    parser.add_argument(
        "--chem_csv",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv"),
        help="Path to chemical properties CSV for auxiliary regression loss (set to empty string to disable).",
    )
    parser.add_argument(
        "--aux_weight",
        type=float,
        default=0.1,
        help="Weight of auxiliary property regression loss relative to link-prediction loss.",
    )
    parser.set_defaults(skip_literals=False, add_inverse=True)
    parser.add_argument(
        "--skip_literals",
        dest="skip_literals",
        action="store_true",
        help="Skip literal tails so only entity-to-entity triples are used.",
    )
    parser.add_argument(
        "--include_literals",
        dest="skip_literals",
        action="store_false",
        help="Include literal tails as entities.",
    )
    parser.add_argument(
        "--add_inverse",
        dest="add_inverse",
        action="store_true",
        help="Add inverse relations and inverse triples.",
    )
    parser.add_argument(
        "--no_inverse",
        dest="add_inverse",
        action="store_false",
        help="Do not add inverse triples.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_mof_uris(ttl_path: str) -> List[str]:
    logging.info("Extracting MOF URIs...")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    query = """
    PREFIX : <http://emmo.info/domain-mof/mof-ontology#>
    SELECT DISTINCT ?mof
    WHERE {
        ?mof a ?type .
        FILTER(?type IN (:MOF, :ExperimentalMOF, :HypotheticalMOF))
    }
    """
    mofs = [str(row.mof) for row in g.query(query)]
    logging.info("Found %d MOFs in TTL query.", len(mofs))
    return mofs


def load_rdf_triples(ttl_path: str, skip_literals: bool) -> List[Tuple[str, str, str]]:
    logging.info("Loading RDF triples from %s", ttl_path)
    g = Graph()
    g.parse(ttl_path, format="turtle")
    
    triples: List[Tuple[str, str, str]] = []
    skipped_literals = 0
    for s, p, o in g:
        if skip_literals and isinstance(o, rdflib.Literal):
            skipped_literals += 1
            continue
        triples.append((str(s), str(p), str(o)))

    logging.info(
        "Collected %d triples (skipped %d literal-tail triples).",
        len(triples),
        skipped_literals,
    )
    return triples


def build_kg_tensors(
    triples: Sequence[Tuple[str, str, str]],
    add_inverse: bool,
    val_ratio: float = 0.01,
) -> Tuple[
    Dict[str, int],
    List[str],
    Dict[str, int],
    List[str],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}

    def get_ent_id(entity: str) -> int:
        if entity not in ent2id:
            ent2id[entity] = len(ent2id)
        return ent2id[entity]

    def get_rel_id(relation: str) -> int:
        if relation not in rel2id:
            rel2id[relation] = len(rel2id)
        return rel2id[relation]

    indexed_triples: List[Tuple[int, int, int]] = []
    inverse_added = 0

    for h, r, t in triples:
        h_id = get_ent_id(h)
        t_id = get_ent_id(t)
        r_id = get_rel_id(r)
        indexed_triples.append((h_id, r_id, t_id))

        if add_inverse:
            inv_rel = f"{r}__inverse"
            inv_r_id = get_rel_id(inv_rel)
            indexed_triples.append((t_id, inv_r_id, h_id))
            inverse_added += 1

    if not indexed_triples:
        raise ValueError("No triples available after preprocessing.")

    triple_tensor = torch.tensor(indexed_triples, dtype=torch.long)
    
    # Split into train and validation
    num_triples = triple_tensor.size(0)
    num_val = int(num_triples * val_ratio)
    perm = torch.randperm(num_triples)
    val_indices = perm[:num_val]
    train_indices = perm[num_val:]
    
    val_triples = triple_tensor[val_indices]
    train_triples = triple_tensor[train_indices]

    # Edge index for message passing should only use train triples
    edge_src = train_triples[:, 0].tolist()
    edge_dst = train_triples[:, 2].tolist()
    edge_type = train_triples[:, 1].tolist()
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

    id2ent = [None] * len(ent2id)
    for key, value in ent2id.items():
        id2ent[value] = key

    id2rel = [None] * len(rel2id)
    for key, value in rel2id.items():
        id2rel[value] = key

    logging.info(
        "KG tensors ready: %d entities, %d relations, %d train triples, %d val triples.",
        len(ent2id),
        len(rel2id),
        train_triples.size(0),
        val_triples.size(0),
    )
    return ent2id, id2ent, rel2id, id2rel, edge_index, edge_type_tensor, train_triples, val_triples


# ---------------------------------------------------------------------------
# ANCHOR REGRESSION HEADS  (auxiliary property-prediction loss)
# ---------------------------------------------------------------------------

_ANCHOR_COLS = [
    "Density",
    "Largest cavity diameter",
    "Pore limiting diameter",
    "Unit cell volume",
    "Band gap (PBE)",
]


def _safe_name(col: str) -> str:
    """Convert a property column name to a valid Python/nn.Module key."""
    return col.replace(" ", "_").replace("(", "").replace(")", "")


class AnchorRegressors(nn.Module):
    """One linear head per anchor property – jointly trained with the GNN.

    Having these extra regression objectives gives the GNN a direct gradient
    signal to embed structural/electronic properties into the MOF vectors,
    which is exactly what was missing when only the link-prediction loss was used.
    """

    def __init__(self, emb_dim: int, property_names: List[str]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {_safe_name(n): nn.Linear(emb_dim, 1, bias=True) for n in property_names}
        )

    def compute_loss(
        self,
        ent_emb: torch.Tensor,
        anchor_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Mean MSE across all anchor properties (values are z-scored)."""
        losses = []
        for safe_name, (indices, values) in anchor_data.items():
            if safe_name not in self.heads:
                continue
            pred = self.heads[safe_name](ent_emb[indices]).squeeze(-1)
            losses.append(F.mse_loss(pred, values))
        if not losses:
            return torch.tensor(0.0, device=ent_emb.device)
        return torch.stack(losses).mean()


def load_anchor_data(
    chem_csv_path: str,
    ent2id: Dict[str, int],
    device: torch.device,
    anchor_cols: Optional[List[str]] = None,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], List[str]]:
    """Load and z-score anchor properties from the chemical CSV.

    Returns
    -------
    anchor_data : dict  {safe_name -> (index_tensor, z-scored value tensor)}
    valid_cols  : list  original column names that had >=100 MOFs in the graph
    """
    if anchor_cols is None:
        anchor_cols = _ANCHOR_COLS

    usecols = ["mof_uri"] + [c for c in anchor_cols]
    df = pd.read_csv(chem_csv_path, usecols=usecols)

    anchor_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    valid_cols: List[str] = []

    for col in anchor_cols:
        if col not in df.columns:
            logging.warning("Anchor column '%s' not found in CSV, skipping.", col)
            continue

        sub = df[["mof_uri", col]].dropna()
        indices, values = [], []
        for _, row in sub.iterrows():
            idx = ent2id.get(row["mof_uri"])
            if idx is not None:
                indices.append(idx)
                values.append(float(row[col]))

        if len(indices) < 100:
            logging.warning(
                "Anchor '%s' has only %d graph-mapped MOFs (<100), skipping.", col, len(indices)
            )
            continue

        vals_arr = np.array(values, dtype=np.float32)
        mean, std = float(vals_arr.mean()), float(vals_arr.std())
        if std < 1e-8:
            logging.warning("Anchor '%s' has near-zero std, skipping.", col)
            continue

        vals_norm = (vals_arr - mean) / std
        safe = _safe_name(col)
        anchor_data[safe] = (
            torch.tensor(indices, dtype=torch.long, device=device),
            torch.tensor(vals_norm, dtype=torch.float32, device=device),
        )
        valid_cols.append(col)
        logging.info(
            "Anchor '%s' (%s): %d samples  [mean=%.4f  std=%.4f]",
            col, safe, len(indices), mean, std,
        )

    if not anchor_data:
        raise ValueError(
            "No anchor properties loaded. Check --chem_csv path and that MOF URIs overlap."
        )

    return anchor_data, valid_cols


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

class CompGCNConv(MessagePassing):
    def __init__(self, emb_dim: int, num_relations: int, comp_op: str, dropout: float, num_bases: int = -1) -> None:
        super().__init__(aggr="add", flow="source_to_target")
        self.emb_dim = emb_dim
        self.num_relations = num_relations
        self.comp_op = comp_op
        self.dropout = dropout
        self.num_bases = num_bases

        self.w_in = nn.Parameter(torch.empty(emb_dim, emb_dim))
        self.w_loop = nn.Parameter(torch.empty(emb_dim, emb_dim))
        self.w_rel = nn.Parameter(torch.empty(emb_dim, emb_dim))
        self.loop_rel = nn.Parameter(torch.empty(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))
        self.layer_norm = nn.LayerNorm(emb_dim)

        if self.num_bases > 0:
            self.basis = nn.Parameter(torch.empty(num_bases, emb_dim, emb_dim))
            self.rel_weights = nn.Parameter(torch.empty(num_relations, num_bases))
        else:
            self.rel_weights = nn.Parameter(torch.empty(num_relations, emb_dim, emb_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_in)
        nn.init.xavier_uniform_(self.w_loop)
        nn.init.xavier_uniform_(self.w_rel)
        nn.init.xavier_uniform_(self.loop_rel.unsqueeze(0))
        nn.init.zeros_(self.bias)
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.basis)
            nn.init.xavier_uniform_(self.rel_weights)
        else:
            nn.init.xavier_uniform_(self.rel_weights)

    def _compose(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.comp_op == "sub":
            return x - r
        if self.comp_op == "mult":
            return x * r
        # Circular correlation
        x_fft = torch.fft.rfft(x, dim=-1)
        r_fft = torch.fft.rfft(r, dim=-1)
        corr = torch.fft.irfft(torch.conj(x_fft) * r_fft, n=x.size(-1), dim=-1)
        return corr.real + 1e-12

    def forward(
        self,
        ent_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute relation-specific weights via basis decomposition
        if self.num_bases > 0:
            w_rel_all = torch.einsum("rb, bxy -> rxy", self.rel_weights, self.basis)
        else:
            w_rel_all = self.rel_weights

        out = self.propagate(edge_index, x=ent_emb, rel_emb=rel_emb, edge_type=edge_type, w_rel_all=w_rel_all)
        out = out @ self.w_in

        loop_msg = self._compose(ent_emb, self.loop_rel.unsqueeze(0).expand_as(ent_emb))
        loop_msg = loop_msg @ self.w_loop
        out = (out + loop_msg) * 0.5
        out = self.layer_norm(out + self.bias)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        rel_out = rel_emb @ self.w_rel
        rel_out = F.relu(rel_out)
        rel_out = F.dropout(rel_out, p=self.dropout, training=self.training)
        return out, rel_out

    def message(
        self,
        x_j: torch.Tensor,
        rel_emb: torch.Tensor,
        edge_type: torch.Tensor,
        w_rel_all: torch.Tensor,
    ) -> torch.Tensor:
        rel_vec = rel_emb[edge_type]
        msg = self._compose(x_j, rel_vec)
        # Apply relation-specific weights per-relation to avoid [9M, 256, 256] tensor
        out = torch.zeros_like(msg)
        for r_id in range(self.num_relations):
            mask = edge_type == r_id
            if mask.any():
                out[mask] = msg[mask] @ w_rel_all[r_id]
        return out


class CompGCNModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        emb_dim: int,
        num_layers: int,
        dropout: float,
        comp_op: str,
        decoder: str,
        num_bases: int = -1,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.layers = nn.ModuleList(
            [CompGCNConv(emb_dim=emb_dim, num_relations=num_relations, comp_op=comp_op, dropout=dropout, num_bases=num_bases) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient checkpointing.
        
        Checkpointing does NOT store intermediate activations during the forward
        pass; instead it recomputes them one layer at a time during backward.
        This reduces peak memory from O(num_layers * num_edges * emb_dim)
        to O(1 * num_edges * emb_dim), at the cost of ~2x forward compute.
        """
        ent = x
        rel = self.relation_emb.weight
        for layer in self.layers:
            # Gradient checkpointing: trade compute for memory
            ent, rel = ckpt_fn(
                layer,
                ent,
                rel,
                edge_index,
                edge_type,
                use_reentrant=False,
            )
        return ent, rel

    def score_triples(self, triples: torch.Tensor, ent: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        h = ent[triples[:, 0]]
        r = rel[triples[:, 1]]
        t = ent[triples[:, 2]]
        if self.decoder == "distmult":
            return (h * r * t).sum(dim=-1)
        # TransE-style score (higher is better).
        return -(h + r - t).abs().sum(dim=-1)


# ---------------------------------------------------------------------------
# NEGATIVE SAMPLING
# ---------------------------------------------------------------------------

def generate_negative_triples(
    pos_triples: torch.Tensor,
    num_entities: int,
    device: torch.device,
) -> torch.Tensor:
    neg = pos_triples.clone()
    batch_size = neg.size(0)
    random_entities = torch.randint(0, num_entities, (batch_size,), device=device)
    corrupt_head = torch.rand(batch_size, device=device) < 0.5
    neg[corrupt_head, 0] = random_entities[corrupt_head]
    neg[~corrupt_head, 2] = random_entities[~corrupt_head]
    return neg


def generate_hard_negatives(
    pos_triples: torch.Tensor,
    num_entities: int,
    device: torch.device,
    triples_by_rel: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """
    Type-aware negative sampling. For a triple (h, r, t), 
    pick a negative tail t' that has appeared as a tail for relation r elsewhere.
    """
    neg = pos_triples.clone()
    batch_size = neg.size(0)
    corrupt_head = torch.rand(batch_size, device=device) < 0.5
    
    for r_id in torch.unique(pos_triples[:, 1]):
        r_mask = pos_triples[:, 1] == r_id
        num_r = r_mask.sum().item()
        
        possible_triples = triples_by_rel.get(r_id.item())
        if possible_triples is None:
            continue
        
        rand_idx = torch.randint(0, possible_triples.size(0), (num_r,), device=device)
        
        r_corrupt_head = corrupt_head & r_mask
        r_corrupt_tail = (~corrupt_head) & r_mask
        
        if r_corrupt_head.any():
            neg[r_corrupt_head, 0] = possible_triples[rand_idx[r_corrupt_head[r_mask]], 0]
        if r_corrupt_tail.any():
            neg[r_corrupt_tail, 2] = possible_triples[rand_idx[r_corrupt_tail[r_mask]], 2]
                
    return neg


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mrr(
    model: CompGCNModel,
    val_triples: torch.Tensor,
    ent_emb: torch.Tensor,
    rel_emb: torch.Tensor,
    num_entities: int,
    device: torch.device,
    batch_size: int = 100,
) -> Dict[str, float]:
    """
    Calculate MRR and Hits@K on a validation set.
    Ranks against 2000 random entities for speed.
    """
    model.eval()
    mrrs = []
    hits1 = []
    hits3 = []
    hits10 = []
    
    # Subsample validation for quick logging
    if val_triples.size(0) > 1000:
        indices = torch.randperm(val_triples.size(0), device=val_triples.device)[:1000]
        val_triples = val_triples[indices]

    for i in range(val_triples.size(0)):
        h, r, t = val_triples[i]
        
        num_samples = min(num_entities, 2000)
        neg_entities = torch.randint(0, num_entities, (num_samples,), device=device)
        neg_entities[0] = t  # ensure true tail is present
        
        test_triples = torch.stack([
            torch.full((num_samples,), h, device=device),
            torch.full((num_samples,), r, device=device),
            neg_entities
        ], dim=1)
        
        scores = model.score_triples(test_triples, ent_emb, rel_emb)
        
        _, sorted_idx = torch.sort(scores, descending=True)
        rank = (sorted_idx == 0).nonzero(as_tuple=True)[0].item() + 1
        
        mrrs.append(1.0 / rank)
        hits1.append(1.0 if rank <= 1 else 0.0)
        hits3.append(1.0 if rank <= 3 else 0.0)
        hits10.append(1.0 if rank <= 10 else 0.0)
        
    return {
        "mrr": sum(mrrs) / len(mrrs),
        "hits@1": sum(hits1) / len(hits1),
        "hits@3": sum(hits3) / len(hits3),
        "hits@10": sum(hits10) / len(hits10),
    }


# ---------------------------------------------------------------------------
# TRAINING LOOP  –  gradient checkpointing + edge dropout
# ---------------------------------------------------------------------------

def train_model(
    model: CompGCNModel,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    train_triples: torch.Tensor,
    val_triples: torch.Tensor,
    num_entities: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    margin: float,
    edge_dropout: float,
    num_sub_batches: int,
    log_every: int,
    checkpoint_every: int,
    output_dir: str,
    anchor_regressors: Optional[AnchorRegressors] = None,
    anchor_data: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    aux_weight: float = 0.1,
    start_epoch: int = 1,
) -> Tuple[List[float], float]:
    # Include auxiliary regression head parameters so they are updated together
    all_params = list(model.parameters())
    if anchor_regressors is not None:
        all_params += list(anchor_regressors.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    # Cosine LR decay: starts at `lr`, anneals to lr/100 over `epochs` total steps.
    # Create scheduler fresh then fast-forward to the correct position for resume.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,
    )
    for _ in range(start_epoch - 1):
        scheduler.step()
    bce_loss = nn.BCEWithLogitsLoss()
    losses: List[float] = []
    best_mrr = 0.0
    best_loss = float("inf")

    # Group triples by relation for hard negative sampling
    triples_by_rel: Dict[int, torch.Tensor] = {}
    for r_id in range(model.relation_emb.num_embeddings):
        mask = train_triples[:, 1] == r_id
        if mask.any():
            triples_by_rel[r_id] = train_triples[mask]

    num_edges = edge_index.size(1)
    num_triples = train_triples.size(0)
    
    logging.info(
        "Training config: %d epochs, batch_size=%d, edge_dropout=%.2f, "
        "sub_batches=%d, lr=%.1e, checkpointing=ON",
        epochs, batch_size, edge_dropout, num_sub_batches, lr,
    )
    logging.info(
        "Memory budget: full graph has %d edges. After %.0f%% edge dropout -> ~%d edges per epoch.",
        num_edges, edge_dropout * 100, int(num_edges * (1 - edge_dropout)),
    )
    if start_epoch > 1:
        logging.info("Resuming from epoch %d / %d", start_epoch, epochs)

    wall_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.time()

        # ------------------------------------------------------------------
        # 1. Edge dropout: randomly keep (1 - edge_dropout) fraction of edges.
        #    This reduces peak memory during backprop AND acts as regularisation.
        # ------------------------------------------------------------------
        if edge_dropout > 0.0:
            keep_mask = torch.rand(num_edges, device=device) > edge_dropout
            ei = edge_index[:, keep_mask]
            et = edge_type[keep_mask]
        else:
            ei = edge_index
            et = edge_type

        # ------------------------------------------------------------------
        # 2. ONE full-graph forward pass with gradient checkpointing.
        #    Checkpointing means PyTorch does NOT store intermediate activations
        #    for each layer; it recomputes them one-at-a-time during backward.
        #    Peak memory: O(1 layer * edges * emb_dim) instead of O(all layers).
        # ------------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        ent, rel = model(model.entity_emb.weight, ei, et)

        # ------------------------------------------------------------------
        # 3. Score MULTIPLE sub-batches of triples against these embeddings.
        #    The scoring is cheap (just indexing + dot products); the expensive
        #    part was the GNN forward, which we only did once.
        # ------------------------------------------------------------------
        total_loss = torch.tensor(0.0, device=device)
        for _ in range(num_sub_batches):
            perm = torch.randperm(num_triples, device=device)[:batch_size]
            pos_batch = train_triples[perm]

            # Hard negatives 50% of the time
            if torch.rand(1).item() < 0.5:
                neg_batch = generate_hard_negatives(pos_batch, num_entities, device, triples_by_rel)
            else:
                neg_batch = generate_negative_triples(pos_batch, num_entities, device)

            pos_score = model.score_triples(pos_batch, ent, rel)
            neg_score = model.score_triples(neg_batch, ent, rel)

            if model.decoder == "distmult":
                logits = torch.cat([pos_score, neg_score], dim=0)
                labels = torch.cat([
                    torch.ones_like(pos_score),
                    torch.zeros_like(neg_score),
                ], dim=0)
                total_loss = total_loss + bce_loss(logits, labels)
            else:
                total_loss = total_loss + F.relu(margin - pos_score + neg_score).mean()

        total_loss = total_loss / num_sub_batches

        # ------------------------------------------------------------------
        # 3b. Auxiliary property regression loss.
        #     This gives the GNN a direct gradient signal to encode structural/
        #     electronic properties (density, pore sizes, band gap…) into MOF
        #     entity embeddings, on top of the link-prediction objective.
        # ------------------------------------------------------------------
        if anchor_regressors is not None and anchor_data:
            anchor_regressors.train()
            aux_loss = anchor_regressors.compute_loss(ent, anchor_data)
            total_loss = total_loss + aux_weight * aux_loss

        # ------------------------------------------------------------------
        # 4. Backward: checkpointing recomputes each layer's intermediates
        #    one at a time, keeping peak memory at ~1 layer worth.
        # ------------------------------------------------------------------
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        avg_loss = total_loss.item()
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start

        # ------------------------------------------------------------------
        # 5. Logging & Validation
        # ------------------------------------------------------------------
        if epoch % log_every == 0 or epoch == 1:
            with torch.no_grad():
                model.eval()
                # Use FULL graph (no dropout) for validation embedding
                ent_val, rel_val = model(model.entity_emb.weight, edge_index, edge_type)
                metrics = evaluate_mrr(model, val_triples, ent_val, rel_val, num_entities, device)

            elapsed_h = (time.time() - wall_start) / 3600
            eta_h = elapsed_h / (epoch - start_epoch + 1) * (epochs - epoch) if epoch > start_epoch else 0
            current_lr = scheduler.get_last_lr()[0]
            logging.info(
                "Epoch %d/%d [%.1fs] - Loss: %.6f | MRR: %.4f | Hits@10: %.4f | "
                "LR: %.2e | Elapsed: %.2fh | ETA: %.2fh",
                epoch, epochs, epoch_time, avg_loss,
                metrics["mrr"], metrics["hits@10"],
                current_lr, elapsed_h, eta_h,
            )

            if metrics["mrr"] > best_mrr:
                best_mrr = metrics["mrr"]
                best_path = os.path.join(output_dir, "compgcn_best_model.pt")
                try:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "avg_loss": avg_loss,
                            "mrr": metrics["mrr"],
                        },
                        best_path,
                    )
                    logging.info("New best MRR: %.4f (Saved)", best_mrr)
                except Exception as save_exc:
                    logging.warning("Could not save best model (epoch %d): %s", epoch, save_exc)

        if epoch % checkpoint_every == 0:
            # Save model state only (no optimizer state) to keep file size ~1.2G not ~3.5G.
            # A transient filesystem error must NOT kill the training run.
            ckpt_path = os.path.join(output_dir, f"compgcn_checkpoint_epoch_{epoch}.pt")
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "avg_loss": avg_loss,
                    },
                    ckpt_path,
                )
                logging.info("Saved checkpoint: %s", ckpt_path)
            except Exception as save_exc:
                logging.warning("Checkpoint save failed (epoch %d): %s — continuing.", epoch, save_exc)

        if avg_loss < best_loss:
            best_loss = avg_loss

    total_time = (time.time() - wall_start) / 3600
    logging.info("Training finished in %.2f hours.", total_time)
    return losses, best_loss


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

def export_mof_embeddings(
    mof_uris: Sequence[str],
    ent2id: Dict[str, int],
    ent_emb: torch.Tensor,
    output_path: str,
) -> int:
    rows = []
    for mof_uri in mof_uris:
        idx = ent2id.get(mof_uri)
        if idx is None:
            continue
        vec = ent_emb[idx].detach().cpu().tolist()
        row = {"mof_uri": mof_uri}
        row.update({f"emb_{i}": float(v) for i, v in enumerate(vec)})
        rows.append(row)

    if not rows:
        raise ValueError("No MOF embeddings to export. Check URI overlap with entity mapping.")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return len(rows)


def export_relation_embeddings(rel2id: Dict[str, int], rel_emb: torch.Tensor, output_path: str) -> int:
    rows = []
    for rel_uri, rel_id in rel2id.items():
        vec = rel_emb[rel_id].detach().cpu().tolist()
        row = {"relation_uri": rel_uri}
        row.update({f"rel_emb_{i}": float(v) for i, v in enumerate(vec)})
        rows.append(row)

    if not rows:
        raise ValueError("No relation embeddings to export.")

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return len(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    mof_uris = extract_mof_uris(args.ttl_path)
    triples = load_rdf_triples(args.ttl_path, skip_literals=args.skip_literals)
    ent2id, id2ent, rel2id, id2rel, edge_index, edge_type, train_triples, val_triples = build_kg_tensors(
        triples,
        add_inverse=args.add_inverse,
        val_ratio=args.val_ratio,
    )

    valid_mof_count = sum(1 for uri in mof_uris if uri in ent2id)
    logging.info("MOF coverage in graph: %d/%d", valid_mof_count, len(mof_uris))
    if valid_mof_count == 0:
        raise ValueError("No MOF URIs found in entity mapping.")

    # ------------------------------------------------------------------
    # Auxiliary regression heads (optional – enabled when --chem_csv given)
    # ------------------------------------------------------------------
    anchor_regressors: Optional[AnchorRegressors] = None
    anchor_data: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None

    if args.chem_csv:
        logging.info("Loading anchor property data from %s", args.chem_csv)
        try:
            anchor_data, valid_cols = load_anchor_data(
                args.chem_csv, ent2id, device
            )
            anchor_regressors = AnchorRegressors(args.emb_dim, valid_cols).to(device)
            logging.info(
                "Auxiliary regression enabled: %d properties  (aux_weight=%.3f)",
                len(valid_cols), args.aux_weight,
            )
        except Exception as exc:
            logging.warning("Could not load anchor data (%s); training without aux loss.", exc)

    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    train_triples = train_triples.to(device)
    val_triples = val_triples.to(device)

    model = CompGCNModel(
        num_entities=len(ent2id),
        num_relations=len(rel2id),
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        comp_op=args.comp_op,
        decoder=args.decoder,
        num_bases=args.num_bases,
    ).to(device)

    logging.info(
        "Model: %d params | %d entities | %d relations | %d bases | decoder=%s | comp_op=%s",
        sum(p.numel() for p in model.parameters()),
        len(ent2id), len(rel2id), args.num_bases, args.decoder, args.comp_op,
    )

    # ------------------------------------------------------------------
    # Optional resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume:
        logging.info("Loading checkpoint for resume: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        resumed_epoch = ckpt.get("epoch", args.start_epoch - 1)
        logging.info("Resumed model from epoch %d", resumed_epoch)

    losses, best_loss = train_model(
        model=model,
        edge_index=edge_index,
        edge_type=edge_type,
        train_triples=train_triples,
        val_triples=val_triples,
        num_entities=len(ent2id),
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
        edge_dropout=args.edge_dropout,
        num_sub_batches=args.num_sub_batches,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        anchor_regressors=anchor_regressors,
        anchor_data=anchor_data,
        aux_weight=args.aux_weight,
        start_epoch=args.start_epoch,
    )
    
    with torch.no_grad():
        model.eval()
        # Final embedding extraction — full graph, no gradients → no autograd overhead
        ent_emb, rel_emb = model(model.entity_emb.weight, edge_index, edge_type)

    mof_out = os.path.join(
        args.output_dir,
        f"mof_compgcn_embeddings_{args.emb_dim}d_{args.num_layers}layers.csv",
    )
    rel_out = os.path.join(
        args.output_dir,
        f"relation_compgcn_embeddings_{args.emb_dim}d_{args.num_layers}layers.csv",
    )

    num_mofs_exported = export_mof_embeddings(mof_uris, ent2id, ent_emb, mof_out)
    num_rels_exported = export_relation_embeddings(rel2id, rel_emb, rel_out)

    torch.save(ent2id, os.path.join(args.output_dir, "ent2id.pt"))
    torch.save(rel2id, os.path.join(args.output_dir, "rel2id.pt"))
    torch.save(id2ent, os.path.join(args.output_dir, "id2ent.pt"))
    torch.save(id2rel, os.path.join(args.output_dir, "id2rel.pt"))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "best_loss": best_loss,
            "last_loss": losses[-1] if losses else None,
        },
        os.path.join(args.output_dir, "compgcn_final_model.pt"),
    )

    logging.info("Saved MOF embeddings: %s (%d rows)", mof_out, num_mofs_exported)
    logging.info("Saved relation embeddings: %s (%d rows)", rel_out, num_rels_exported)

    # Visualize training loss
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('CompGCN Training Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(args.output_dir, "compgcn_training_loss.png")
        plt.savefig(loss_plot_path)
        logging.info("Saved training loss plot: %s", loss_plot_path)

    logging.info(
        "Training done. Final loss: %.6f | Best loss: %.6f",
        losses[-1] if losses else float("nan"),
        best_loss,
    )


if __name__ == "__main__":
    main()
