#!/usr/bin/env python3
"""
generate_topology_tsne.py
=========================
Generate t-SNE visualizations for CompGCN, Node2Vec, and TransE embeddings,
colored by topology. Compute manifold quality metrics (silhouette score, kNN purity)
to identify which embedding method produces the best topology separation.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPC = os.path.join(BASE, "MOFKG_from_hpc")
CHEM_PATH = os.path.join(HPC, "studies", "data", "chemcial_properties.csv")
OUT_DIR = os.path.join(BASE, "paper", "figures", "generated")
RESULTS_DIR = os.path.join(HPC, "studies", "results", "concept_vectors")

# Embedding paths
COMPGCN_CSV = os.path.join(HPC, "studies", "data", "gnn_embeddings", 
                           "mof_compgcn_embeddings_256d_3layers.csv")
TRANSE_CSV = os.path.join(HPC, "studies", "data", "transe_embeddings",
                          "mof_transe_embeddings_256d.csv")
NODE2VEC_PT = os.path.join(HPC, "studies", "data", "node2vec",
                           "mof_embeddings_256d_p1.0_q1.0.pt")

SEED = 42
MAX_SAMPLES = 8000  # Sample size for t-SNE
PLASMA = plt.cm.plasma

# Styling
sns.set_theme(style="whitegrid", palette="plasma")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_embedding_matrix(method: str):
    """Load (uris, matrix) for a given method name."""
    print(f"Loading {method} embeddings...")
    
    if method == "Node2Vec":
        if not os.path.exists(NODE2VEC_PT):
            return None, None
        saved = torch.load(NODE2VEC_PT, map_location="cpu", weights_only=False)
        emb = saved["embeddings"]
        if hasattr(emb, "numpy"):
            emb = emb.numpy()
        else:
            emb = np.asarray(emb)
        ent2id = saved["ent2id"]
        uris = [None] * len(ent2id)
        for uri, idx in ent2id.items():
            uris[idx] = uri
        return uris, emb
    
    elif method == "CompGCN":
        csv_path = COMPGCN_CSV
    elif method == "TransE":
        csv_path = TRANSE_CSV
    else:
        return None, None
    
    if not os.path.exists(csv_path):
        return None, None
    
    df = pd.read_csv(csv_path)
    if "mof_uri" not in df.columns:
        df = df.rename(columns={df.columns[0]: "mof_uri"})
    
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        emb_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df["mof_uri"].tolist(), df[emb_cols].to_numpy(dtype=np.float32)


def compute_manifold_quality(uris, emb, labels_df, label_col="topology", 
                             max_samples=5000, min_class_size=20, seed=42):
    """
    Compute silhouette score and kNN purity for an embedding method.
    Returns (silhouette, knn_purity, n_samples, perplexity_used).
    """
    print(f"  Computing manifold quality for {label_col}...")
    
    # Create URI to index mapping
    uri2idx = {u: i for i, u in enumerate(uris)}
    
    # Merge with labels
    sub = labels_df[labels_df["mof_uri"].isin(uri2idx)].dropna(subset=[label_col]).copy()
    # Filter out ERROR,UNKNOWN topologies
    sub = sub[~sub[label_col].str.contains("ERROR", case=False, na=False)]
    sub = sub[~sub[label_col].str.contains("UNKNOWN", case=False, na=False)]
    
    if sub.empty:
        print(f"    No samples with {label_col} labels")
        return np.nan, np.nan, 0, 0
    
    # Filter to classes with enough samples
    counts = sub[label_col].value_counts()
    keep = counts[counts >= min_class_size].index
    sub = sub[sub[label_col].isin(keep)]
    
    if sub.empty or sub[label_col].nunique() < 2:
        print(f"    Not enough classes with >= {min_class_size} samples")
        return np.nan, np.nan, 0, 0
    
    # Sample if too large
    rng = np.random.default_rng(seed)
    if len(sub) > max_samples:
        sub = sub.sample(n=max_samples, random_state=seed)
    
    # Get embeddings for selected samples
    idx = np.array([uri2idx[u] for u in sub["mof_uri"].values])
    X = emb[idx]
    y = sub[label_col].values
    
    # Compute silhouette score
    try:
        sil = float(silhouette_score(X, y, sample_size=min(len(X), 3000), random_state=seed))
    except Exception as e:
        print(f"    Silhouette computation failed: {e}")
        sil = np.nan
    
    # Compute kNN purity
    k = 10
    try:
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        _, nbrs = nn.kneighbors(X)
        nbrs = nbrs[:, 1:]  # Exclude self
        neighbor_labels = y[nbrs]
        purity = float(np.mean(neighbor_labels == y[:, None]))
    except Exception as e:
        print(f"    kNN purity computation failed: {e}")
        purity = np.nan
    
    # Calculate perplexity
    perplexity = min(30, len(X) - 1, len(X) // 4)
    
    print(f"    Silhouette: {sil:.4f}, kNN purity: {purity:.4f}, n={len(sub)}")
    return sil, purity, int(len(sub)), perplexity


def compute_tsne(X, seed=42, perplexity=30, n_iter=1000):
    """Compute t-SNE projection with PCA preprocessing."""
    print(f"  Running PCA + t-SNE (perplexity={perplexity})...")
    
    # PCA preprocessing
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
    
    # t-SNE
    X_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        n_jobs=-1
    ).fit_transform(X_pca)
    
    return X_2d


def plot_tsne_by_topology(method, uris, emb, labels_df, out_path, 
                         max_samples=8000, top_n_topologies=12, seed=42):
    """Generate t-SNE plot colored by topology."""
    print(f"\nGenerating t-SNE for {method}...")
    
    # Create URI to index mapping
    uri2idx = {u: i for i, u in enumerate(uris)}
    
    # Merge with topology labels and exclude ERROR,UNKNOWN
    sub = labels_df[labels_df["mof_uri"].isin(uri2idx)].dropna(subset=["topology"]).copy()
    # Filter out ERROR,UNKNOWN topologies
    sub = sub[~sub["topology"].str.contains("ERROR", case=False, na=False)]
    sub = sub[~sub["topology"].str.contains("UNKNOWN", case=False, na=False)]
    
    if sub.empty:
        print(f"  No samples with topology labels for {method}")
        return None
    
    # Get top topologies
    topo_counts = sub["topology"].value_counts()
    top_topos = topo_counts.head(top_n_topologies).index.tolist()
    sub["topology_cat"] = sub["topology"].where(
        sub["topology"].isin(top_topos), other="Other"
    )
    
    # Sample if needed
    rng = np.random.default_rng(seed)
    if len(sub) > max_samples:
        sub = sub.sample(n=max_samples, random_state=seed)
    
    # Get embeddings
    idx = np.array([uri2idx[u] for u in sub["mof_uri"].values])
    X = emb[idx]
    
    # Compute t-SNE
    perplexity = min(30, len(X) - 1, len(X) // 4)
    X_2d = compute_tsne(X, seed=seed, perplexity=perplexity)
    
    # Save projection points
    proj_df = pd.DataFrame({
        "mof_uri": sub["mof_uri"].values,
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "topology": sub["topology"].values,
        "topology_cat": sub["topology_cat"].values
    })
    csv_path = os.path.join(RESULTS_DIR, f"topology_projection_{method}.csv")
    proj_df.to_csv(csv_path, index=False)
    print(f"  Saved projection points to {csv_path}")
    
    # Plot
    cats = top_topos + ["Other"]
    cat_colors = {c: PLASMA(i / (len(cats) - 1)) for i, c in enumerate(cats)}
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    for cat in cats:
        mask = proj_df["topology_cat"] == cat
        if mask.sum() == 0:
            continue
        ax.scatter(
            proj_df.loc[mask, "x"],
            proj_df.loc[mask, "y"],
            c=[cat_colors[cat]],
            s=5,
            alpha=0.6,
            label=cat,
            rasterized=True
        )
    
    ax.set_title(f"{method} t-SNE Colored by Topology", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.legend(title="Topology", fontsize=9, title_fontsize=10, 
             markerscale=2, loc="best", ncol=2, framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")
    
    return proj_df


def generate_comparison_figure(quality_df, out_path):
    """Generate comparison figure showing manifold quality metrics."""
    print("\nGenerating comparison figure...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    methods = ["CompGCN", "Node2Vec", "TransE"]
    colors = {"CompGCN": PLASMA(0.15), "Node2Vec": PLASMA(0.55), "TransE": PLASMA(0.85)}
    
    x = np.arange(len(methods))
    width = 0.6
    
    # Silhouette scores
    sil_vals = [quality_df[quality_df["Method"] == m]["Silhouette"].values[0] 
                for m in methods if m in quality_df["Method"].values]
    sil_methods = [m for m in methods if m in quality_df["Method"].values]
    ax1.bar(range(len(sil_methods)), sil_vals, width=width, 
           color=[colors[m] for m in sil_methods], edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(len(sil_methods)))
    ax1.set_xticklabels(sil_methods)
    ax1.set_ylabel("Silhouette Score", fontsize=11)
    ax1.set_title("(a) Topology Cluster Separation", fontsize=12, fontweight="bold")
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    
    # Add value labels
    for i, v in enumerate(sil_vals):
        ax1.text(i, v + 0.01 if v >= 0 else v - 0.01, f"{v:.3f}", 
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    
    # kNN purity
    knn_vals = [quality_df[quality_df["Method"] == m]["kNN_Purity"].values[0] 
                for m in methods if m in quality_df["Method"].values]
    ax2.bar(range(len(sil_methods)), knn_vals, width=width,
           color=[colors[m] for m in sil_methods], edgecolor="white", linewidth=0.5)
    ax2.set_xticks(range(len(sil_methods)))
    ax2.set_xticklabels(sil_methods)
    ax2.set_ylabel("kNN Label Purity (k=10)", fontsize=11)
    ax2.set_title("(b) Local Topology Coherence", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.02)
    
    # Add value labels
    for i, v in enumerate(knn_vals):
        ax2.text(i, v + 0.02, f"{v:.3f}", 
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    fig.suptitle("Topology Manifold Quality Comparison", 
                fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def generate_combined_tsne_figure(methods_data, quality_df, out_path):
    """Generate 3-panel figure showing all three embeddings side by side."""
    print("\nGenerating combined 3-panel figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = ["CompGCN", "Node2Vec", "TransE"]
    
    for ax, method in zip(axes, methods):
        if method not in methods_data:
            ax.text(0.5, 0.5, f"No data for {method}", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(method, fontsize=12, fontweight="bold")
            continue
        
        proj_df = methods_data[method]
        cats = proj_df["topology_cat"].unique()
        cat_colors = {c: PLASMA(i / (len(cats) - 1)) for i, c in enumerate(sorted(cats))}
        
        for cat in sorted(cats):
            mask = proj_df["topology_cat"] == cat
            if mask.sum() == 0:
                continue
            ax.scatter(
                proj_df.loc[mask, "x"],
                proj_df.loc[mask, "y"],
                c=[cat_colors[cat]],
                s=3,
                alpha=0.55,
                label=cat,
                rasterized=True
            )
        
        ax.set_title(method, fontsize=18, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False)
    
    # Shared legend - large, spanning full width
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 7),
                  fontsize=16, markerscale=4, frameon=True, 
                  bbox_to_anchor=(0.5, -0.08), columnspacing=1.5, handletextpad=0.5)
    
    fig.suptitle("t-SNE Projections of KG Embeddings Colored by Topology",
                fontsize=20, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    print("=" * 70)
    print("Topology-Colored t-SNE Visualization")
    print("=" * 70)
    
    # Load chemical properties (includes topology)
    print(f"\nLoading chemical properties from {CHEM_PATH}...")
    chem_df = pd.read_csv(CHEM_PATH, low_memory=False)
    print(f"  Loaded {len(chem_df)} MOF records")
    
    # Check topology column
    if "topology" not in chem_df.columns:
        print("ERROR: No 'topology' column found in chemical properties!")
        return
    
    topo_counts = chem_df["topology"].value_counts()
    print(f"  Found {len(topo_counts)} unique topologies")
    print(f"  Top 10 topologies:\n{topo_counts.head(10)}")
    
    # Load embeddings
    methods = ["CompGCN", "Node2Vec", "TransE"]
    embeddings = {}
    
    for method in methods:
        uris, emb = load_embedding_matrix(method)
        if uris is not None and emb is not None:
            print(f"  {method}: {len(uris)} entities, {emb.shape[1]}D")
            embeddings[method] = (uris, emb)
        else:
            print(f"  {method}: NOT FOUND")
    
    if not embeddings:
        print("\nERROR: No embeddings found!")
        return
    
    # Compute manifold quality for each method
    print("\n" + "=" * 70)
    print("Computing Manifold Quality Metrics")
    print("=" * 70)
    
    quality_rows = []
    for method, (uris, emb) in embeddings.items():
        sil, knn, n, perp = compute_manifold_quality(
            uris, emb, chem_df, label_col="topology",
            max_samples=MAX_SAMPLES, seed=SEED
        )
        quality_rows.append({
            "Method": method,
            "Silhouette": sil,
            "kNN_Purity": knn,
            "N_Samples": n,
            "Perplexity": perp
        })
    
    quality_df = pd.DataFrame(quality_rows)
    quality_csv = os.path.join(RESULTS_DIR, "topology_manifold_quality.csv")
    quality_df.to_csv(quality_csv, index=False)
    print(f"\nSaved quality metrics to {quality_csv}")
    print("\nManifold Quality Summary:")
    print(quality_df.to_string(index=False))
    
    # Identify best methods
    best_sil = quality_df.loc[quality_df["Silhouette"].idxmax()]
    best_knn = quality_df.loc[quality_df["kNN_Purity"].idxmax()]
    print(f"\nBest Silhouette: {best_sil['Method']} ({best_sil['Silhouette']:.4f})")
    print(f"Best kNN Purity: {best_knn['Method']} ({best_knn['kNN_Purity']:.4f})")
    
    # Generate individual t-SNE plots
    print("\n" + "=" * 70)
    print("Generating t-SNE Visualizations")
    print("=" * 70)
    
    methods_data = {}
    for method, (uris, emb) in embeddings.items():
        out_path = os.path.join(OUT_DIR, f"tsne_topology_{method}.png")
        proj_df = plot_tsne_by_topology(
            method, uris, emb, chem_df, out_path,
            max_samples=MAX_SAMPLES, seed=SEED
        )
        if proj_df is not None:
            methods_data[method] = proj_df
    
    # Generate comparison figures
    print("\n" + "=" * 70)
    print("Generating Comparison Figures")
    print("=" * 70)
    
    if quality_df is not None and not quality_df.empty:
        comparison_path = os.path.join(OUT_DIR, "fig_topology_manifold_comparison.png")
        generate_comparison_figure(quality_df, comparison_path)
    
    if methods_data:
        combined_path = os.path.join(OUT_DIR, "fig_supp_tsne_topology.png")
        generate_combined_tsne_figure(methods_data, quality_df, combined_path)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Quality metrics: {quality_csv}")
    print(f"  - Individual plots: {OUT_DIR}/tsne_topology_*.png")
    print(f"  - Comparison: {OUT_DIR}/fig_topology_manifold_comparison.png")
    print(f"  - Combined 3-panel: {OUT_DIR}/fig_supp_tsne_topology.png")


if __name__ == "__main__":
    main()
