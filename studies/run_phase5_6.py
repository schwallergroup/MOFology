#!/usr/bin/env python3
"""
run_phase5_6.py
===============
Standalone script that picks up AFTER the full study pipeline's
Phases 1-4 and runs:
  Phase 5: t-SNE visualisations
  Phase 6: Summary plots & tables

It reloads the saved embeddings from disk — NO retraining or KG parsing.
"""

import os
import sys
import gc
import logging
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ── logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Import CompGCN model architecture (for loading weights) ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings"))
from CompGCN import CompGCNModel  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
# CONFIG  (must match run_full_study.py)
# ═══════════════════════════════════════════════════════════════════════
CHEM_PATH   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv")
COMPGCN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings")
TRANSE_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings")
N2V_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec")
OUT_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/full_comparison)

SEED        = 42
EMB_DIM     = 256
TSNE_MOF_N  = 15_000
TSNE_ENT_N  = 20_000

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


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
        return "Abstract"
    if "Element_" in frag or frag.startswith("MetalElement_"):
        return "Element"
    return "Other"


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDING LOADERS  (from saved artifacts, no KG parsing needed)
# ═══════════════════════════════════════════════════════════════════════

def load_compgcn_embeddings(device: torch.device) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load CompGCN embeddings from the saved final model checkpoint.

    Because CompGCN embeddings are the OUTPUT of the GNN (not the raw
    nn.Embedding weights), we need to re-run a forward pass through the
    saved model.  We rebuild edge_index from the saved ent2id / rel2id
    and the original training triples stored inside the checkpoint.
    """
    log.info("  ── CompGCN: loading saved model ──")

    ent2id: Dict[str, int] = torch.load(
        os.path.join(COMPGCN_DIR, "ent2id.pt"), weights_only=False)
    rel2id: Dict[str, int] = torch.load(
        os.path.join(COMPGCN_DIR, "rel2id.pt"), weights_only=False)

    # Choose best checkpoint
    best_path = os.path.join(COMPGCN_DIR, "compgcn_best_model.pt")
    if not os.path.exists(best_path):
        best_path = os.path.join(COMPGCN_DIR, "best_model.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)

    num_entities  = checkpoint.get("num_entities",  len(ent2id))
    num_relations = checkpoint.get("num_relations", len(rel2id))
    emb_dim       = checkpoint.get("emb_dim", EMB_DIM)
    num_layers    = checkpoint.get("num_layers", 2)
    dropout       = checkpoint.get("dropout", 0.1)
    comp_op       = checkpoint.get("comp_op", "mult")
    decoder       = checkpoint.get("decoder", "distmult")
    num_bases     = checkpoint.get("num_bases", 4)

    log.info("    Config: %d ent, %d rel, %dd, %d layers, %d bases",
             num_entities, num_relations, emb_dim, num_layers, num_bases)

    # Rebuild edge_index from the KG triples file (saved during training)
    # We need the triples to do a forward pass. Parse from the TTL only once.
    log.info("    Rebuilding edge_index from KG …")
    import rdflib
    from rdflib import Graph as RDFGraph

    rdf_graph = RDFGraph()
    rdf_graph.parse(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl"), format="turtle")

    src_list: List[int] = []
    dst_list: List[int] = []
    et_list:  List[int] = []
    for s, p, o in rdf_graph:
        if not isinstance(o, (rdflib.URIRef, rdflib.BNode)):
            continue
        s_str, p_str, o_str = str(s), str(p), str(o)
        s_id = ent2id.get(s_str)
        o_id = ent2id.get(o_str)
        r_id = rel2id.get(p_str)
        if s_id is None or o_id is None or r_id is None:
            continue
        src_list.append(s_id)
        dst_list.append(o_id)
        et_list.append(r_id)
        inv_r_id = rel2id.get(f"{p_str}__inverse")
        if inv_r_id is not None:
            src_list.append(o_id)
            dst_list.append(s_id)
            et_list.append(inv_r_id)

    del rdf_graph
    gc.collect()

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).to(device)
    edge_type  = torch.tensor(et_list, dtype=torch.long).to(device)
    log.info("    Edge index: %d edges (inc. inverses)", edge_index.size(1))

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

    log.info("    Running forward pass for %d entities …", num_entities)
    with torch.no_grad():
        ent = model.entity_emb.weight
        rel = model.relation_emb.weight
        for layer in model.layers:
            ent, rel = layer(ent, rel, edge_index, edge_type)
        all_emb = ent.cpu()

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


def load_node2vec_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load the 256d Node2Vec embeddings saved by the main pipeline."""
    log.info("  ── Node2Vec: loading saved 256d embeddings ──")
    saved = torch.load(
        os.path.join(N2V_DIR, "mof_embeddings_256d_p1.0_q1.0.pt"),
        weights_only=False)
    all_emb = saved["embeddings"]
    ent2id  = saved["ent2id"]
    log.info("    Node2Vec embeddings: %s", list(all_emb.shape))
    return all_emb, ent2id


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

        np.random.shuffle(selected_indices)
        selected_indices = selected_indices[:n_total]

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

    # ── Load all three sets of embeddings ──
    log.info("=" * 70)
    log.info("Loading saved embeddings (no retraining)")
    log.info("=" * 70)

    compgcn_emb, compgcn_ent2id = load_compgcn_embeddings(device)
    transe_emb,  transe_ent2id  = load_transe_embeddings()
    n2v_emb,     n2v_ent2id     = load_node2vec_embeddings()

    all_entity_embs = {
        "CompGCN":  (compgcn_emb, compgcn_ent2id),
        "TransE":   (transe_emb,  transe_ent2id),
        "Node2Vec": (n2v_emb,     n2v_ent2id),
    }

    # ── Build MOF-only DataFrames ──
    log.info("  Building MOF-only embedding DataFrames …")
    mof_emb_dfs: Dict[str, pd.DataFrame] = {}
    for name, (emb_tensor, ent2id) in all_entity_embs.items():
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

    # ── Phase 5: t-SNE ──
    generate_tsne_visualizations(all_entity_embs, mof_emb_dfs)

    # ── Phase 6: Summary (reload Phase 3+4 CSVs from disk) ──
    lp_csv   = os.path.join(OUT_DIR, "lp_comparison.csv")
    chem_csv = os.path.join(OUT_DIR, "chem_prediction_comparison.csv")

    lp_df   = pd.read_csv(lp_csv)   if os.path.exists(lp_csv)   else pd.DataFrame()
    chem_df = pd.read_csv(chem_csv)  if os.path.exists(chem_csv) else pd.DataFrame()

    if lp_df.empty:
        log.warning("  lp_comparison.csv not found — skipping Phase 6")
    else:
        generate_summary(lp_df, chem_df)

    elapsed_m = (time.time() - wall_start) / 60
    log.info("=" * 70)
    log.info("PHASE 5+6 COMPLETE  —  %.1f minutes elapsed", elapsed_m)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
