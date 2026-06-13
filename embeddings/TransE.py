import argparse
import logging
import os
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import rdflib
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdflib import Graph
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_TTL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TransE baseline model on MOF KG.")
    parser.add_argument("--ttl_path", type=str, default=DEFAULT_TTL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--norm_p", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(skip_literals=True)
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
    for s, p, o in tqdm(g, desc="Parsing triples"):
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
) -> Tuple[
    Dict[str, int],
    List[str],
    Dict[str, int],
    List[str],
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
    for h, r, t in tqdm(triples, desc="Indexing triples"):
        h_id = get_ent_id(h)
        t_id = get_ent_id(t)
        r_id = get_rel_id(r)
        indexed_triples.append((h_id, r_id, t_id))

    triple_tensor = torch.tensor(indexed_triples, dtype=torch.long)

    id2ent = [None] * len(ent2id)
    for key, value in ent2id.items():
        id2ent[value] = key

    id2rel = [None] * len(rel2id)
    for key, value in rel2id.items():
        id2rel[value] = key

    logging.info(
        "KG tensors ready: %d entities, %d relations, %d triples.",
        len(ent2id),
        len(rel2id),
        triple_tensor.size(0),
    )
    return ent2id, id2ent, rel2id, id2rel, triple_tensor

class TransEModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int, norm_p: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.norm_p = norm_p

        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        
        # Normalize relation embeddings initially
        with torch.no_grad():
            self.relation_emb.weight.data = F.normalize(self.relation_emb.weight.data, p=2, dim=1)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        t = self.entity_emb(t_idx)

        # Score is L1 or L2 distance: ||h + r - t||
        score = torch.norm(h + r - t, p=self.norm_p, dim=1)
        return score

    def normalize_entities(self):
        with torch.no_grad():
            self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)

def generate_negative_triples(pos_triples, num_entities, device):
    neg = pos_triples.clone()
    batch_size = neg.size(0)
    random_entities = torch.randint(0, num_entities, (batch_size,), device=device)
    corrupt_head = torch.rand(batch_size, device=device) < 0.5
    neg[corrupt_head, 0] = random_entities[corrupt_head]
    neg[~corrupt_head, 2] = random_entities[~corrupt_head]
    return neg

def train_model(
    model,
    train_triples,
    num_entities,
    device,
    args
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_loss = float("inf")
    losses = []

    num_triples = train_triples.size(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        # TransE standard: normalize entities to unit length before each epoch
        model.normalize_entities()
        
        perm = torch.randperm(num_triples, device=device)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, num_triples, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pos_batch = train_triples[idx]
            neg_batch = generate_negative_triples(pos_batch, num_entities, device)

            optimizer.zero_grad(set_to_none=True)
            
            pos_score = model(pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2])
            neg_score = model(neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])

            # Margin ranking loss: max(0, margin + pos_score - neg_score)
            # Note: for TransE, smaller score is better (distance), so we want neg_score > pos_score
            loss = F.relu(args.margin + pos_score - neg_score).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        losses.append(avg_loss)
        
        # Step the scheduler based on epoch loss
        scheduler.step(avg_loss)
        
        if epoch % args.log_every == 0 or epoch == 1:
            logging.info("Epoch %d/%d - Loss: %.6f", epoch, args.epochs, avg_loss)

        if epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"transe_checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
            }, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "transe_best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "avg_loss": avg_loss,
            }, best_path)

    return losses, best_loss

def export_mof_embeddings(mof_uris, ent2id, ent_emb, output_path):
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
        logging.warning("No MOF embeddings to export.")
        return 0

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return len(rows)

def export_relation_embeddings(rel2id, rel_emb, output_path):
    rows = []
    for rel_uri, rel_id in rel2id.items():
        vec = rel_emb[rel_id].detach().cpu().tolist()
        row = {"relation_uri": rel_uri}
        row.update({f"rel_emb_{i}": float(v) for i, v in enumerate(vec)})
        rows.append(row)

    if not rows:
        return 0

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return len(rows)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    mof_uris = extract_mof_uris(args.ttl_path)
    triples = load_rdf_triples(args.ttl_path, skip_literals=args.skip_literals)
    ent2id, id2ent, rel2id, id2rel, train_triples = build_kg_tensors(triples)

    valid_mof_count = sum(1 for uri in mof_uris if uri in ent2id)
    logging.info("MOF coverage in graph: %d/%d", valid_mof_count, len(mof_uris))

    train_triples = train_triples.to(device)

    model = TransEModel(
        num_entities=len(ent2id),
        num_relations=len(rel2id),
        emb_dim=args.emb_dim,
        norm_p=args.norm_p
    ).to(device)

    losses, best_loss = train_model(
        model=model,
        train_triples=train_triples,
        num_entities=len(ent2id),
        device=device,
        args=args
    )

    # Final normalization
    model.normalize_entities()
    
    ent_emb = model.entity_emb.weight
    rel_emb = model.relation_emb.weight

    mof_out = os.path.join(args.output_dir, f"mof_transe_embeddings_{args.emb_dim}d.csv")
    rel_out = os.path.join(args.output_dir, f"relation_transe_embeddings_{args.emb_dim}d.csv")

    num_mofs_exported = export_mof_embeddings(mof_uris, ent2id, ent_emb, mof_out)
    num_rels_exported = export_relation_embeddings(rel2id, rel_emb, rel_out)

    torch.save(ent2id, os.path.join(args.output_dir, "ent2id.pt"))
    torch.save(rel2id, os.path.join(args.output_dir, "rel2id.pt"))
    torch.save(id2ent, os.path.join(args.output_dir, "id2ent.pt"))
    torch.save(id2rel, os.path.join(args.output_dir, "id2rel.pt"))

    logging.info("Saved MOF embeddings: %s (%d rows)", mof_out, num_mofs_exported)
    logging.info("Saved relation embeddings: %s (%d rows)", rel_out, num_rels_exported)
    logging.info("Training done. Final loss: %.6f | Best loss: %.6f", losses[-1], best_loss)

if __name__ == "__main__":
    main()
