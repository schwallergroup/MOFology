#!/usr/bin/env python3
"""
run_full_study.py
=================
Unified comparison pipeline for Node2Vec, TransE, and CompGCN
on the heterogeneous MOF Knowledge Graph.

Usage:
    python run_full_study.py

Phases:
  1. Parse the TTL knowledge graph (once)
  2. Compute / load all-entity embeddings for each method
  3. Link-prediction downstream evaluation (AUC)
  4. Chemical-property prediction downstream evaluation (R², RMSE)
  5. t-SNE visualizations (crystal system + entity type)
  6. Summary comparison plots and tables
"""

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import os
import sys
import gc
import logging
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import rdflib
from rdflib import Graph, URIRef

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from gensim.models import Word2Vec
from pecanpy import pecanpy as pecanpy_module
from pathlib import Path

# Get project root based on file location
project_root = Path(__file__).parent.parent

# ── logging (set BEFORE CompGCN import so its basicConfig is a no-op) ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Import CompGCN model architecture ──
sys.path.insert(0, str(project_root / "embeddings"))
from CompGCN import CompGCNModel  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
KG_PATH     = str(project_root.parent / "data" / "kg" / "mof_kg.ttl")
CHEM_PATH   = str(project_root / "studies" / "data" / "chemcial_properties.csv")
COMPGCN_DIR = str(project_root / "embeddings" / "data" / "gnn_embeddings")
TRANSE_DIR  = str(project_root / "embeddings" / "data" / "transe_embeddings")
N2V_DIR     = str(project_root / "embeddings" / "data" / "node2vec")
OUT_DIR     = str(project_root / "studies" / "results" / "full_comparison")

SEED        = 42
EMB_DIM     = 256
LP_SAMPLES  = 50_000
TSNE_MOF_N  = 15_000
TSNE_ENT_N  = 20_000

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Relations filtered out of the EVALUATION graph (inverse/redundant)
RELATIONS_TO_DROP = {
    "http://emmo.info/domain-mof/mof-ontology#usedInMOF",
    "http://emmo.info/domain-mof/mof-ontology#isComponentOf",
    "http://emmo.info/domain-mof/mof-ontology#hasPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasComputationalPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasStructuralPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasPhysicalPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#describedIn",
    "http://emmo.info/domain-mof/mof-ontology#describedInAbstract",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
}


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def infer_entity_type(uri: str) -> str:
    """Classify a KG entity URI into a human-readable type."""
    frag = uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
    if frag.startswith("MOF_") or frag.startswith("FuncMOF_"):
        return "MOF"
    if frag.startswith("LINKER_"):
        return "Linker"
    if frag.startswith("CLUSTER_"):
        return "MetalCluster"
    if frag.startswith("PROP_") or frag.startswith("LATTICE_"):
        return "Property"
    if frag.startswith("Topology_"):
        return "Topology"
    if frag.startswith("SpaceGroup_") or frag.startswith("CrystalSystem_"):
        return "Structural"
    if frag.startswith("ABSTRACT_"):
        return "Publication"
    if frag.startswith("SYN_") or frag.startswith("FUNC_"):
        return "Synthesis"
    if "synthesis#" in uri:
        return "Synthesis"
    if "owl#" in uri or "rdf-syntax" in uri or "rdf-schema" in uri:
        return "Ontology"
    return "Other"


def build_embedding_matrix(
    method_emb: torch.Tensor,
    method_ent2id: Dict[str, int],
    eval_node_to_idx: Dict[str, int],
    emb_dim: int,
) -> Tuple[torch.Tensor, int]:
    """Map embeddings from a method's entity-ID space to the evaluation
    graph's node-index space.  Nodes without an embedding remain zero."""
    x = torch.zeros(len(eval_node_to_idx), emb_dim, dtype=torch.float32)
    matched = 0
    for uri, eval_idx in eval_node_to_idx.items():
        method_idx = method_ent2id.get(uri)
        if method_idx is not None and method_idx < method_emb.size(0):
            x[eval_idx] = method_emb[method_idx].cpu().float()
            matched += 1
    return x, matched


def check_required_files():
    """Verify that all required input files exist before starting."""
    required = [
        KG_PATH,
        CHEM_PATH,
        os.path.join(COMPGCN_DIR, "compgcn_final_model.pt"),
        os.path.join(COMPGCN_DIR, "ent2id.pt"),
        os.path.join(COMPGCN_DIR, "rel2id.pt"),
        os.path.join(TRANSE_DIR, "transe_best_model.pt"),
        os.path.join(TRANSE_DIR, "ent2id.pt"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        for f in missing:
            log.error("Missing file: %s", f)
        raise FileNotFoundError(f"{len(missing)} required file(s) missing — see log above.")
    log.info("All required input files present.")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — PARSE KG
# ═══════════════════════════════════════════════════════════════════════

def parse_kg(ttl_path: str):
    """Parse TTL once.  Returns rdflib Graph plus evaluation-graph tensors."""
    log.info("=" * 70)
    log.info("PHASE 1: Parsing Knowledge Graph")
    log.info("=" * 70)
    t0 = time.time()

    rdf_graph = Graph()
    rdf_graph.parse(ttl_path, format="turtle")
    log.info("  Parsed %d raw triples in %.1f min",
             len(rdf_graph), (time.time() - t0) / 60)

    # Extract ALL entity→entity triples (skip literals)
    all_triples: List[Tuple[str, str, str]] = []
    for s, p, o in rdf_graph:
        if isinstance(o, rdflib.Literal):
            continue
        all_triples.append((str(s), str(p), str(o)))
    log.info("  %d entity-entity triples (literals skipped)", len(all_triples))

    # Build EVALUATION graph (filtered relations)
    eval_nodes: set = set()
    eval_triples: List[Tuple[str, str, str]] = []
    for s, p, o in all_triples:
        if (p in RELATIONS_TO_DROP
                or "Owner" in p
                or "usedIn" in p
                or "isComponent" in p):
            continue
        eval_nodes.add(s)
        eval_nodes.add(o)
        eval_triples.append((s, p, o))

    eval_node_to_idx = {n: i for i, n in enumerate(sorted(eval_nodes))}
    eval_rel_to_idx  = {r: i for i, r in enumerate(
        sorted({t[1] for t in eval_triples}))}

    src    = [eval_node_to_idx[s] for s, _, _ in eval_triples]
    dst    = [eval_node_to_idx[o] for _, _, o in eval_triples]
    etypes = [eval_rel_to_idx[p]  for _, p, _ in eval_triples]

    eval_edge_index = torch.tensor([src, dst], dtype=torch.long)
    eval_edge_type  = torch.tensor(etypes, dtype=torch.long)

    log.info("  Eval graph: %d nodes, %d edges, %d relation types",
             len(eval_node_to_idx), eval_edge_index.size(1),
             len(eval_rel_to_idx))
    log.info("  Phase 1 complete in %.1f min", (time.time() - t0) / 60)

    return (rdf_graph, all_triples,
            eval_node_to_idx, eval_rel_to_idx,
            eval_edge_index, eval_edge_type)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def load_compgcn_embeddings(
    all_triples: List[Tuple[str, str, str]],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load trained CompGCN, rebuild edge_index, run one forward pass
    to get GNN-processed embeddings for ALL 1.2 M entities."""
    log.info("  ── CompGCN: loading saved model ──")

    ent2id: Dict[str, int] = torch.load(
        os.path.join(COMPGCN_DIR, "ent2id.pt"), weights_only=False)
    rel2id: Dict[str, int] = torch.load(
        os.path.join(COMPGCN_DIR, "rel2id.pt"), weights_only=False)

    checkpoint = torch.load(
        os.path.join(COMPGCN_DIR, "compgcn_final_model.pt"),
        map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})

    num_entities  = len(ent2id)
    num_relations = len(rel2id)
    emb_dim    = args.get("emb_dim", 256)
    num_layers = args.get("num_layers", 2)
    dropout    = args.get("dropout", 0.2)
    comp_op    = args.get("comp_op", "mult")
    decoder    = args.get("decoder", "distmult")
    num_bases  = args.get("num_bases", 4)

    log.info("    Config: %d ent, %d rel, %dd, %d layers, %d bases",
             num_entities, num_relations, emb_dim, num_layers, num_bases)

    # ── Rebuild edge_index from KG using saved mappings ──
    log.info("    Rebuilding edge_index …")
    src_list: List[int] = []
    dst_list: List[int] = []
    et_list:  List[int] = []
    for s, p, o in all_triples:
        s_id = ent2id.get(s)
        o_id = ent2id.get(o)
        r_id = rel2id.get(p)
        if s_id is None or o_id is None or r_id is None:
            continue
        src_list.append(s_id)
        dst_list.append(o_id)
        et_list.append(r_id)
        # inverse edge
        inv_r_id = rel2id.get(f"{p}__inverse")
        if inv_r_id is not None:
            src_list.append(o_id)
            dst_list.append(s_id)
            et_list.append(inv_r_id)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).to(device)
    edge_type  = torch.tensor(et_list, dtype=torch.long).to(device)
    log.info("    Edge index: %d edges (inc. inverses)", edge_index.size(1))

    # ── Instantiate model & load weights ──
    model = CompGCNModel(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=emb_dim,
        num_layers=num_layers,
        dropout=dropout,
        comp_op=comp_op,
        decoder=decoder,
        num_bases=num_bases,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Forward pass (bypass checkpointing for inference) ──
    log.info("    Running forward pass for %d entities …", num_entities)
    with torch.no_grad():
        ent = model.entity_emb.weight
        rel = model.relation_emb.weight
        for layer in model.layers:
            ent, rel = layer(ent, rel, edge_index, edge_type)
        all_emb = ent.cpu()

    # Free GPU memory
    del model, edge_index, edge_type
    torch.cuda.empty_cache()
    gc.collect()

    log.info("    CompGCN embeddings: %s", list(all_emb.shape))
    return all_emb, ent2id


def load_transe_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load TransE entity embeddings directly from saved model weights."""
    log.info("  ── TransE: loading saved model ──")

    ent2id: Dict[str, int] = torch.load(
        os.path.join(TRANSE_DIR, "ent2id.pt"), weights_only=False)
    checkpoint = torch.load(
        os.path.join(TRANSE_DIR, "transe_best_model.pt"),
        map_location="cpu", weights_only=False)
    all_emb = checkpoint["model_state_dict"]["entity_emb.weight"]

    log.info("    TransE embeddings: %s", list(all_emb.shape))
    return all_emb, ent2id


def train_node2vec_256d(
    rdf_graph: Graph,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Re-train Node2Vec at 256 d using pecanpy (fast C++ walks)."""
    log.info("  ── Node2Vec: training at %dd ──", EMB_DIM)
    t0 = time.time()

    # ── 1. Build integer edgelist ──
    node_to_id: Dict[str, int] = {}
    id_to_node: Dict[int, str] = {}
    current_id = 0

    def _get_id(node_str: str) -> int:
        nonlocal current_id
        if node_str not in node_to_id:
            node_to_id[node_str] = current_id
            id_to_node[current_id] = node_str
            current_id += 1
        return node_to_id[node_str]

    edgelist_path = os.path.join(OUT_DIR, "_temp_edgelist.tsv")
    edge_count = 0
    with open(edgelist_path, "w") as f:
        for s, p, o in rdf_graph:
            if isinstance(o, (rdflib.URIRef, rdflib.BNode)):
                s_str, o_str = str(s), str(o)
                if s_str == o_str:
                    continue
                f.write(f"{_get_id(s_str)}\t{_get_id(o_str)}\n")
                edge_count += 1

    log.info("    Edgelist: %d edges, %d nodes (%.1f min)",
             edge_count, len(node_to_id), (time.time() - t0) / 60)

    # ── 2. pecanpy random walks ──
    log.info("    Running pecanpy (p=1.0, q=1.0, walks=5, length=10) …")
    g = pecanpy_module.SparseOTF(p=1.0, q=1.0, workers=32, extend=True)
    g.read_edg(edgelist_path, weighted=False, directed=False, delimiter="\t")
    walks = g.simulate_walks(num_walks=5, walk_length=10)
    log.info("    %d walks generated (%.1f min)",
             len(walks), (time.time() - t0) / 60)

    # ── 3. Word2Vec ──
    log.info("    Training Word2Vec at %dd …", EMB_DIM)
    w2v = Word2Vec(
        sentences=walks,
        vector_size=EMB_DIM,
        window=10,
        min_count=1,
        sg=1,           # skip-gram
        workers=32,
    )
    log.info("    Vocabulary: %d nodes", len(w2v.wv))

    # ── 4. Build tensor + mapping ──
    n2v_ent2id: Dict[str, int] = {}
    emb_list: List[torch.Tensor] = []
    for uri, nid in node_to_id.items():
        nid_str = str(nid)
        if nid_str in w2v.wv:
            n2v_ent2id[uri] = len(emb_list)
            emb_list.append(torch.tensor(w2v.wv[nid_str], dtype=torch.float32))

    all_emb = torch.stack(emb_list) if emb_list else torch.zeros(0, EMB_DIM)

    # ── 5. Save 256d Node2Vec embeddings for future use ──
    n2v_save_path = os.path.join(N2V_DIR, "mof_embeddings_256d_p1.0_q1.0.pt")
    torch.save({"embeddings": all_emb, "ent2id": n2v_ent2id}, n2v_save_path)
    log.info("    Saved Node2Vec 256d embeddings → %s", n2v_save_path)

    # Cleanup
    del g, walks, w2v
    gc.collect()
    if os.path.exists(edgelist_path):
        os.remove(edgelist_path)

    log.info("    Node2Vec embeddings: %s (%.1f min)",
             list(all_emb.shape), (time.time() - t0) / 60)
    return all_emb, n2v_ent2id


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — LINK PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_link_prediction(
    embeddings: Dict[str, torch.Tensor],
    eval_edge_index: torch.Tensor,
    num_nodes: int,
) -> pd.DataFrame:
    """Downstream link-prediction evaluation with sklearn classifiers.
    Features: concat(h_emb, t_emb) — identical for every method."""
    log.info("=" * 70)
    log.info("PHASE 3: Link Prediction Evaluation")
    log.info("=" * 70)

    num_samples = min(LP_SAMPLES, eval_edge_index.size(1))
    perm = torch.randperm(eval_edge_index.size(1))[:num_samples]

    pos_h = eval_edge_index[0, perm]
    pos_t = eval_edge_index[1, perm]
    neg_t = torch.randint(0, num_nodes, (num_samples,))

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_jobs=-1, random_state=SEED, eval_metric="logloss",
            verbosity=0),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=500, random_state=SEED),
    }

    all_results: List[dict] = []

    for emb_name, x in embeddings.items():
        log.info("  %s (dim=%d) …", emb_name, x.size(1))

        X_pos = torch.cat([x[pos_h], x[pos_t]], dim=1).numpy()
        X_neg = torch.cat([x[pos_h], x[neg_t]], dim=1).numpy()
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * num_samples + [0] * num_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)

        for clf_name, clf_template in classifiers.items():
            clf = clone(clf_template)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            log.info("    %-22s AUC = %.4f", clf_name, auc)
            all_results.append({
                "Embedding": emb_name,
                "Classifier": clf_name,
                "AUC": auc,
            })

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — CHEMICAL PROPERTY PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_chemical_properties(
    mof_emb_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Downstream chemical-property regression using sklearn regressors."""
    log.info("=" * 70)
    log.info("PHASE 4: Chemical Property Prediction")
    log.info("=" * 70)

    df_prop = pd.read_csv(CHEM_PATH)
    log.info("  Properties file: %d MOFs, %d columns", *df_prop.shape)

    metadata_cols = {
        "mof_uri", "csd_code", "chemical_formula", "mofid",
        "topology", "metal_cluster_elements", "linker_smiles",
        "space_group", "crystal_system",
    }

    regressors = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(
            n_jobs=-1, random_state=SEED, verbosity=0),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(512, 256), max_iter=1000,
            early_stopping=True, random_state=SEED),
    }

    all_results: List[dict] = []

    for emb_name, df_emb in mof_emb_dfs.items():
        log.info("  %s (%d MOFs) …", emb_name, len(df_emb))
        df = pd.merge(df_prop, df_emb, on="mof_uri", how="inner")
        if df.empty:
            log.warning("    No overlap — skipping")
            continue
        log.info("    Merged: %d MOFs", len(df))

        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        target_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in metadata_cols and c not in emb_cols
        ]

        for target in target_cols:
            y_col = df[target]
            mask = y_col.notna()
            if mask.sum() < 100:
                continue

            X = df.loc[mask, emb_cols].values.astype(np.float32)
            y_vals = y_col[mask].values.astype(np.float64)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_vals, test_size=0.2, random_state=SEED)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            for reg_name, reg_template in regressors.items():
                reg = clone(reg_template)
                reg.fit(X_train_s, y_train)
                preds = reg.predict(X_test_s)
                r2   = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                all_results.append({
                    "Embedding": emb_name,
                    "Target": target,
                    "Model": reg_name,
                    "R2": r2,
                    "RMSE": rmse,
                })

        log.info("    %s done — %d result rows so far", emb_name, len(all_results))

    return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — t-SNE VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════

def _plot_tsne(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    max_legend: int = 12,
):
    """Create a publication-quality t-SNE scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = sorted(set(labels))

    # Collapse rare labels → "Other" for readability
    if len(unique) > max_legend:
        counts = Counter(labels)
        top = [k for k, _ in counts.most_common(max_legend - 1)]
        labels = np.array(["Other" if l not in top else l for l in labels])
        unique = sorted(set(labels))

    cmap = plt.cm.get_cmap("tab20", max(len(unique), 1))
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap(i)], label=lab, alpha=0.45, s=4,
                   edgecolors="none", rasterized=True)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=4, fontsize=8, loc="best", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("    Saved %s", out_path)


def _run_tsne(X: np.ndarray, seed: int = SEED) -> np.ndarray:
    """PCA→50 d then t-SNE→2 d."""
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
    X_2d  = TSNE(
        n_components=2, perplexity=30, max_iter=1000,
        random_state=seed, init="pca", learning_rate="auto",
    ).fit_transform(X_pca)
    return X_2d


def generate_tsne_visualizations(
    all_entity_embs: Dict[str, Tuple[torch.Tensor, Dict[str, int]]],
    mof_emb_dfs: Dict[str, pd.DataFrame],
):
    """
    Figure Set A: MOF embeddings coloured by crystal system (3 panels)
    Figure Set B: All-entity embeddings coloured by entity type (3 panels)
    """
    log.info("=" * 70)
    log.info("PHASE 5: t-SNE Visualisations")
    log.info("=" * 70)

    # ── Figure Set A: MOF embeddings by crystal system ──
    log.info("  Set A: MOF embeddings by crystal system")
    df_meta = pd.read_csv(CHEM_PATH, usecols=["mof_uri", "crystal_system"])
    df_meta = df_meta.dropna(subset=["crystal_system"])
    if len(df_meta) > TSNE_MOF_N:
        df_meta = df_meta.sample(n=TSNE_MOF_N, random_state=SEED)
    subsample_uris = set(df_meta["mof_uri"])
    crystal_map = dict(zip(df_meta["mof_uri"], df_meta["crystal_system"]))

    for name, df_emb in mof_emb_dfs.items():
        df_sub = df_emb[df_emb["mof_uri"].isin(subsample_uris)].copy()
        if len(df_sub) < 100:
            log.warning("    %s: only %d MOFs matched — skipping", name, len(df_sub))
            continue
        emb_cols = [c for c in df_sub.columns if c.startswith("emb_")]
        X = df_sub[emb_cols].values.astype(np.float32)
        labels = np.array([crystal_map.get(u, "unknown") for u in df_sub["mof_uri"]])

        log.info("    %s: running t-SNE on %d MOFs …", name, len(X))
        X_2d = _run_tsne(X)
        _plot_tsne(
            X_2d, labels,
            f"{name}: MOF Embeddings by Crystal System",
            os.path.join(OUT_DIR, f"tsne_mof_{name.lower()}.png"),
        )

    # ── Figure Set B: All entity embeddings by entity type ──
    log.info("  Set B: All-entity embeddings by entity type")
    for name, (emb_tensor, ent2id) in all_entity_embs.items():
        uri_list   = list(ent2id.keys())
        type_array = np.array([infer_entity_type(u) for u in uri_list])

        # Stratified subsample keeping proportions
        n_total = min(TSNE_ENT_N, len(uri_list))
        type_counts = Counter(type_array)
        selected_indices: List[int] = []
        for t, cnt in type_counts.items():
            frac = cnt / len(uri_list)
            n_pick = max(20, int(frac * n_total))
            candidates = np.where(type_array == t)[0]
            chosen = np.random.choice(
                candidates, min(n_pick, len(candidates)), replace=False)
            selected_indices.extend(chosen.tolist())

        # Trim to budget
        np.random.shuffle(selected_indices)
        selected_indices = selected_indices[:n_total]

        # Gather embeddings
        ent2id_vals = list(ent2id.values())
        idx_tensor = [ent2id_vals[i] for i in selected_indices]
        X = emb_tensor[idx_tensor].numpy().astype(np.float32)
        labels = type_array[selected_indices]

        log.info("    %s: running t-SNE on %d entities …", name, len(X))
        X_2d = _run_tsne(X)
        _plot_tsne(
            X_2d, labels,
            f"{name}: All Entity Embeddings by Type",
            os.path.join(OUT_DIR, f"tsne_entity_{name.lower()}.png"),
            max_legend=10,
        )


# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — SUMMARY PLOTS & TABLES
# ═══════════════════════════════════════════════════════════════════════

def generate_summary(lp_df: pd.DataFrame, chem_df: pd.DataFrame):
    """Comparison bar plots, heatmaps, and a summary table."""
    log.info("=" * 70)
    log.info("PHASE 6: Summary Plots and Tables")
    log.info("=" * 70)

    # ── LP bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=lp_df, x="Classifier", y="AUC", hue="Embedding", ax=ax)
    ax.set_title("Link Prediction: AUC by Classifier and Embedding Method")
    ax.set_ylim(0.5, 1.0)
    ax.legend(title="Embedding")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lp_comparison_barplot.png"), dpi=300)
    plt.close(fig)
    log.info("  Saved lp_comparison_barplot.png")

    # ── Chem prediction heatmap ──
    if not chem_df.empty:
        # Mean R² across all targets for each (Embedding, Model)
        mean_r2 = (chem_df
                   .groupby(["Embedding", "Model"])["R2"]
                   .mean()
                   .reset_index())
        pivot = mean_r2.pivot(index="Model", columns="Embedding", values="R2")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
        ax.set_title("Chemical Property Prediction: Mean R² (all targets)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "chem_prediction_heatmap.png"), dpi=300)
        plt.close(fig)
        log.info("  Saved chem_prediction_heatmap.png")

        # Top-5 targets by average R² — grouped bar chart
        target_avg = (chem_df.groupby("Target")["R2"]
                      .mean()
                      .sort_values(ascending=False))
        top5 = target_avg.head(5).index.tolist()
        chem_top = chem_df[chem_df["Target"].isin(top5)]

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=chem_top, x="Target", y="R2", hue="Embedding", ax=ax)
        ax.set_title("Chemical Property Prediction: Top-5 Targets by R²")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "chem_top5_barplot.png"), dpi=300)
        plt.close(fig)
        log.info("  Saved chem_top5_barplot.png")

    # ── Aggregated summary table ──
    rows: Dict[str, dict] = {}
    for name in lp_df["Embedding"].unique():
        sub = lp_df[lp_df["Embedding"] == name]
        rows[name] = {
            "Method": name,
            "LP_Mean_AUC": round(sub["AUC"].mean(), 4),
            "LP_Best_AUC": round(sub["AUC"].max(), 4),
        }
    if not chem_df.empty:
        for name in chem_df["Embedding"].unique():
            sub = chem_df[chem_df["Embedding"] == name]
            if name not in rows:
                rows[name] = {"Method": name}
            rows[name]["Chem_Mean_R2"]  = round(sub["R2"].mean(), 4)
            rows[name]["Chem_Best_R2"]  = round(sub["R2"].max(), 4)
            rows[name]["Chem_Mean_RMSE"] = round(sub["RMSE"].mean(), 4)

    summary_df = pd.DataFrame(list(rows.values()))
    summary_df.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    log.info("  Summary table:\n%s", summary_df.to_string(index=False))
    log.info("  All outputs in %s", OUT_DIR)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    wall_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Output dir: %s", device, OUT_DIR)
    check_required_files()

    # ── Phase 1: Parse KG ──
    (rdf_graph, all_triples,
     eval_node_to_idx, eval_rel_to_idx,
     eval_edge_index, eval_edge_type) = parse_kg(KG_PATH)

    # ── Phase 2: Embeddings ──
    log.info("=" * 70)
    log.info("PHASE 2: Computing / Loading Embeddings")
    log.info("=" * 70)

    compgcn_emb, compgcn_ent2id = load_compgcn_embeddings(all_triples, device)
    transe_emb,  transe_ent2id  = load_transe_embeddings()
    n2v_emb,     n2v_ent2id     = train_node2vec_256d(rdf_graph)

    # Free rdflib graph (large)
    del rdf_graph
    gc.collect()

    # Map every method's embeddings into the evaluation graph's node-index space
    log.info("  Mapping embeddings to evaluation graph …")
    compgcn_eval, m1 = build_embedding_matrix(
        compgcn_emb, compgcn_ent2id, eval_node_to_idx, compgcn_emb.size(1))
    transe_eval,  m2 = build_embedding_matrix(
        transe_emb,  transe_ent2id,  eval_node_to_idx, transe_emb.size(1))
    n2v_eval,     m3 = build_embedding_matrix(
        n2v_emb,     n2v_ent2id,     eval_node_to_idx, EMB_DIM)

    log.info("    CompGCN  matched %d / %d eval nodes (%.1f%%)",
             m1, len(eval_node_to_idx), 100 * m1 / len(eval_node_to_idx))
    log.info("    TransE   matched %d / %d eval nodes (%.1f%%)",
             m2, len(eval_node_to_idx), 100 * m2 / len(eval_node_to_idx))
    log.info("    Node2Vec matched %d / %d eval nodes (%.1f%%)",
             m3, len(eval_node_to_idx), 100 * m3 / len(eval_node_to_idx))

    eval_embeddings = {
        "CompGCN":  compgcn_eval,
        "TransE":   transe_eval,
        "Node2Vec": n2v_eval,
    }

    # ── Phase 3: Link Prediction ──
    lp_df = evaluate_link_prediction(
        eval_embeddings, eval_edge_index, len(eval_node_to_idx))
    lp_df.to_csv(os.path.join(OUT_DIR, "lp_comparison.csv"), index=False)

    # ── Phase 4: Chemical Property Prediction ──
    # Build MOF-only DataFrames (emb_ columns + mof_uri) for each method
    log.info("  Building MOF-only embedding DataFrames …")
    mof_emb_dfs: Dict[str, pd.DataFrame] = {}
    for name, (emb_tensor, ent2id) in [
        ("CompGCN",  (compgcn_emb, compgcn_ent2id)),
        ("TransE",   (transe_emb,  transe_ent2id)),
        ("Node2Vec", (n2v_emb,     n2v_ent2id)),
    ]:
        rows = []
        for uri, idx in ent2id.items():
            frag = uri.split("#")[-1] if "#" in uri else ""
            if frag.startswith("MOF_") or frag.startswith("FuncMOF_"):
                vec = emb_tensor[idx].numpy()
                row = {"mof_uri": uri}
                row.update({f"emb_{i}": float(v) for i, v in enumerate(vec)})
                rows.append(row)
        mof_emb_dfs[name] = pd.DataFrame(rows)
        log.info("    %s: %d MOF embeddings", name, len(rows))

    chem_df = evaluate_chemical_properties(mof_emb_dfs)
    chem_df.to_csv(os.path.join(OUT_DIR, "chem_prediction_comparison.csv"),
                   index=False)

    # ── Phase 5: t-SNE ──
    all_entity_embs = {
        "CompGCN":  (compgcn_emb, compgcn_ent2id),
        "TransE":   (transe_emb,  transe_ent2id),
        "Node2Vec": (n2v_emb,     n2v_ent2id),
    }
    generate_tsne_visualizations(all_entity_embs, mof_emb_dfs)

    # ── Phase 6: Summary ──
    generate_summary(lp_df, chem_df)

    elapsed_h = (time.time() - wall_start) / 3600
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE  —  %.2f hours elapsed", elapsed_h)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
