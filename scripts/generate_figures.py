#!/usr/bin/env python3
"""
generate_figures.py — Generate all figures for the MOFology paper.
All figures use the seaborn 'plasma' palette.
Reads only existing CSV data — no experiments are re-run.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════
# Global Style Configuration
# ═══════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="plasma")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

PLASMA = plt.cm.plasma
PLASMA_DISCRETE = [PLASMA(i / 6) for i in range(7)]

# ═══════════════════════════════════════════════════════════
# Path Configuration
# ═══════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPC = os.path.join(BASE, "MOFKG_from_hpc")
STUDIES = os.path.join(HPC, "studies", "results")
OUT = os.path.join(BASE, "paper", "figures", "generated")
TABLES = os.path.join(BASE, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)


def save_fig(fig, name):
    """Save figure as PNG only."""
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)
    print(f"  [OK] {name}.png")


# ═══════════════════════════════════════════════════════════
# Fig 1 — Ontology Class Hierarchy
# ═══════════════════════════════════════════════════════════
def fig_ontology_hierarchy():
    """Draw a clean, publication-quality ontology tree with key MOFology concepts."""
    print("Fig 1: Ontology hierarchy (curated tree)...")
    import networkx as nx

    # Curated hierarchy: key concepts organized by domain
    # Structure: {parent: [children]} - focused on most important classes
    hierarchy = {
        "EMMO": ["MOF", "Material", "Process", "Property", "Capability"],
        "MOF": ["ExperimentalMOF", "HypotheticalMOF", "FunctionalizedMOF"],
        "Material": ["MetalCluster", "OrganicLinker", "Topology"],
        "Process": ["SynthesisProcess", "Functionalization"],
        "Property": ["StructuralProperty", "ComputationalProperty"],
        "Capability": ["CO2CaptureCapability", "DACCapability", "GasStorageCapability"],
    }

    # Build directed graph
    G = nx.DiGraph()
    for parent, children in hierarchy.items():
        for child in children:
            G.add_edge(parent, child)

    # Manual tree positions for clean layout
    # Format: {node: (x, y)} - organized in levels with generous spacing
    pos = {
        # Level 0 (root)
        "EMMO": (0, 3),
        # Level 1 - main categories (5 branches)
        "MOF": (-8, 2), "Material": (-4, 2), "Process": (0, 2), "Property": (4, 2), "Capability": (8.5, 2),
        # Level 2 - subcategories (spread out for readability)
        "ExperimentalMOF": (-10.5, 1), "HypotheticalMOF": (-8, 1), "FunctionalizedMOF": (-5.5, 1),
        "MetalCluster": (-4.5, 1), "OrganicLinker": (-3, 1), "Topology": (-1.5, 1),
        "SynthesisProcess": (-0.5, 1), "Functionalization": (0.8, 1),
        "StructuralProperty": (3, 1), "ComputationalProperty": (5, 1),
        "CO2CaptureCapability": (7, 1), "DACCapability": (9, 1), "GasStorageCapability": (11, 1),
    }

    # Compute depth for coloring
    depth = {"EMMO": 0}
    for node in nx.bfs_tree(G, "EMMO"):
        if node != "EMMO":
            depth[node] = nx.shortest_path_length(G, "EMMO", node)

    max_depth = max(depth.values())
    node_colors = [PLASMA(depth.get(n, 0) / max_depth) for n in G.nodes()]

    # Create figure - wide format for tree
    fig, ax = plt.subplots(figsize=(18, 7))

    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#666666",
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        alpha=0.7,
        width=1.5,
        connectionstyle="arc3,rad=0.1"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=900,
        alpha=0.95,
        edgecolors="white",
        linewidths=1.5
    )

    # Clean labels with readable abbreviations
    label_map = {
        "EMMO": "EMMO\n(root)",
        "MOF": "MOF",
        "Material": "Material",
        "Process": "Process",
        "Property": "Property",
        "Capability": "Capability",
        "ExperimentalMOF": "Experimental",
        "HypotheticalMOF": "Hypothetical",
        "FunctionalizedMOF": "Functionalized",
        "MetalCluster": "Metal\nCluster",
        "OrganicLinker": "Organic\nLinker",
        "Topology": "Topology",
        "SynthesisProcess": "Synthesis",
        "Functionalization": "Function-\nalization",
        "StructuralProperty": "Structural",
        "ComputationalProperty": "Computational",
        "CO2CaptureCapability": "CO2\nCapture",
        "DACCapability": "DAC",
        "GasStorageCapability": "Gas\nStorage",
    }
    labels = {n: label_map.get(n, n) for n in G.nodes()}

    # Draw labels below nodes for readability
    label_pos = {node: (x, y - 0.25) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G, label_pos, labels=labels, ax=ax,
        font_size=9,
        font_weight="bold",
        font_color="black",
        verticalalignment="top"
    )

    ax.set_title("MOFology Ontology: Key Class Hierarchy", fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    ax.set_xlim(-12.5, 13)
    ax.set_ylim(0.2, 3.6)

    # Add legend for depth
    sm = ScalarMappable(cmap=PLASMA, norm=Normalize(vmin=0, vmax=max_depth))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.015, pad=0.02, shrink=0.6)
    cbar.set_label("Hierarchy Depth", fontsize=10)
    cbar.set_ticks([0, 1, 2])

    plt.tight_layout()
    save_fig(fig, "fig01_ontology_hierarchy")


def _fig_ontology_manual():
    """Fallback: manually draw a simplified ontology hierarchy."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define hierarchy manually from the TTL we read
    hierarchy = {
        "MOF": ["ExperimentalMOF", "HypotheticalMOF", "FunctionalizedMOF"],
        "MetalCluster": [],
        "OrganicLinker": [],
        "Topology": [],
        "Chemical": ["MetalPrecursor", "LinkerPrecursor", "Solvent", "Additive"],
        "MaterialProperty": ["StructuralProperty", "ComputationalProperty", "PhysicalProperty"],
        "Capability": ["CO2Capture", "DACCapability", "CH4Storage",
                        "H2Storage", "Photocatalytic", "Luminescent"],
        "ApplicationProcess": ["CatalysisProcess", "GasStorage", "SensingProcess"],
        "SynthesisProcess": ["SynthesisCondition", "SynthesisProcedure"],
        "Functionalization": ["AmineFunctionalization", "MetalSubstitution",
                               "LinkerModification", "Grafting"],
        "Publication": ["Abstract"],
        "DataProvenance": [],
    }

    # Build graph
    import networkx as nx
    G = nx.DiGraph()
    for parent, children in hierarchy.items():
        for child in children:
            G.add_edge(parent, child)

    # Compute depth
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    depth = {}
    for root in roots:
        for node in nx.bfs_tree(G, root):
            d = nx.shortest_path_length(G, root, node)
            depth[node] = max(depth.get(node, 0), d)
    for n in G.nodes():
        if n not in depth:
            depth[n] = 0

    max_depth = max(depth.values()) if depth else 1
    node_colors = [PLASMA(depth.get(n, 0) / max_depth) for n in G.nodes()]

    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    clean = {n: n.replace("_", " ") for n in G.nodes()}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True,
                           arrowsize=12, alpha=0.6, width=1.2)
    nx.draw_networkx_labels(G, pos, labels=clean, ax=ax, font_size=7, font_weight="bold")

    ax.set_title("MOFology Ontology Class Hierarchy", fontsize=14, fontweight="bold")
    ax.axis("off")

    sm = ScalarMappable(cmap=PLASMA, norm=Normalize(vmin=0, vmax=max_depth))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Hierarchy Depth", fontsize=10)

    save_fig(fig, "fig01_ontology_hierarchy")


# ═══════════════════════════════════════════════════════════
# Fig 2 — Property Coverage Bar Chart
# ═══════════════════════════════════════════════════════════
def fig_property_coverage():
    print("Fig 2: Property coverage...")
    df = pd.read_csv(os.path.join(HPC, "results", "ML_Chem",
                                   "prediction_results", "property_coverage_report.csv"))
    df = df.sort_values("non_null_count", ascending=True).tail(25)

    norm = Normalize(vmin=df["coverage"].min(), vmax=df["coverage"].max())
    colors = [PLASMA(norm(c)) for c in df["coverage"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df["property"], df["non_null_count"], color=colors, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Number of MOFs with Property (log scale)", fontsize=11)
    ax.set_title(f"Property Coverage Across {df['total_samples'].iloc[0]:,} MOFs with Computed Properties", fontsize=12, fontweight="bold")
    ax.set_xscale("log")

    save_fig(fig, "fig02_property_coverage")


# ═══════════════════════════════════════════════════════════
# Fig 3 — t-SNE Embedding Comparison (3-panel)
# ═══════════════════════════════════════════════════════════
def _load_embedding_matrix(method: str):
    """Load (uris, matrix) for a given method name."""
    import torch
    if method == "Node2Vec":
        pt_path = os.path.join(HPC, "studies", "data", "node2vec",
                               "mof_embeddings_256d_p1.0_q1.0.pt")
        if not os.path.exists(pt_path):
            return None, None
        saved = torch.load(pt_path, map_location="cpu", weights_only=False)
        emb = saved["embeddings"].numpy() if hasattr(saved["embeddings"], "numpy") else np.asarray(saved["embeddings"])
        ent2id = saved["ent2id"]
        uris = [None] * len(ent2id)
        for uri, idx in ent2id.items():
            uris[idx] = uri
        return uris, emb
    if method == "CompGCN":
        csv_path = os.path.join(HPC, "studies", "data", "gnn_embeddings",
                                "mof_compgcn_embeddings_256d_3layers.csv")
    elif method == "TransE":
        csv_path = os.path.join(HPC, "studies", "data", "transe_embeddings",
                                "mof_transe_embeddings_256d.csv")
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


def _compute_structure_quality(method: str, labels_df: pd.DataFrame, label_col: str,
                                max_samples: int = 5000, seed: int = 0):
    """Compute silhouette score and kNN-label purity for an embedding method
    under a given categorical label column. Returns (silhouette, knn_purity, n)."""
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    uris, emb = _load_embedding_matrix(method)
    if uris is None:
        return np.nan, np.nan, 0
    uri2idx = {u: i for i, u in enumerate(uris)}
    sub = labels_df[labels_df["mof_uri"].isin(uri2idx)].dropna(subset=[label_col]).copy()
    if sub.empty:
        return np.nan, np.nan, 0
    counts = sub[label_col].value_counts()
    keep = counts[counts >= 20].index
    sub = sub[sub[label_col].isin(keep)]
    if sub.empty or sub[label_col].nunique() < 2:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    if len(sub) > max_samples:
        sub = sub.sample(n=max_samples, random_state=seed)
    idx = np.asarray([uri2idx[u] for u in sub["mof_uri"].values])
    X = emb[idx]
    y = sub[label_col].values
    try:
        sil = float(silhouette_score(X, y, sample_size=min(len(X), 3000), random_state=seed))
    except Exception:
        sil = np.nan
    k = 10
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    _, nbrs = nn.kneighbors(X)
    nbrs = nbrs[:, 1:]
    neighbor_labels = y[nbrs]
    purity = float(np.mean(neighbor_labels == y[:, None]))
    return sil, purity, int(len(sub))


def _primary_metal(raw):
    """Extract the first metal symbol from semicolon-separated metal_cluster_elements."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    # Prefer a single-element symbol (length 1 or 2); skip compound strings like 'CuBa'
    for p in parts:
        if 1 <= len(p) <= 2 and p[0].isupper():
            return p
    return parts[0] if parts else None


def fig_tsne_embeddings():
    """Fig 3 — Structure quality of KG embeddings (silhouette + kNN purity)."""
    print("Fig 3: Embedding structure quality (silhouette / kNN purity)...")
    chem_path = os.path.join(HPC, "studies", "data", "chemcial_properties.csv")
    if not os.path.exists(chem_path):
        print("  [SKIP] chemical properties not found")
        return
    chem_df = pd.read_csv(chem_path, low_memory=False)
    if "metal_element" not in chem_df.columns and "metal_cluster_elements" in chem_df.columns:
        chem_df["metal_element"] = chem_df["metal_cluster_elements"].apply(_primary_metal)

    def _source(uri):
        if not isinstance(uri, str):
            return "Other"
        if "FuncMOF_" in uri:
            return "Functionalized"
        if "MOF_STAB_" in uri:
            return "Hypothetical"
        if "MOF_qmof-" in uri:
            return "QMOF"
        if "MOF_" in uri:
            return "CSD"
        return "Other"

    chem_df["source"] = chem_df["mof_uri"].apply(_source)
    label_specs = [
        ("metal_element", "Metal element"),
        ("topology", "Topology"),
        ("source", "MOF source"),
    ]
    methods = ["CompGCN", "Node2Vec", "TransE"]
    rows = []
    for method in methods:
        for col, _ in label_specs:
            if col not in chem_df.columns:
                continue
            sil, pur, n = _compute_structure_quality(method, chem_df, col)
            rows.append({"Method": method, "Label": col, "Silhouette": sil, "kNNPurity": pur, "N": n})
    quality = pd.DataFrame(rows)
    if quality.empty:
        print("  [SKIP] No structure-quality rows computed")
        return
    quality.to_csv(os.path.join(STUDIES, "concept_vectors", "embedding_structure_quality.csv"), index=False)

    fig, (ax_sil, ax_knn) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"CompGCN": PLASMA(0.15), "Node2Vec": PLASMA(0.55), "TransE": PLASMA(0.85)}
    label_names = [l for l, _ in label_specs]
    label_display = {c: lbl for c, lbl in label_specs}
    x = np.arange(len(label_names))
    w = 0.27
    for i, m in enumerate(methods):
        vals_sil = [quality[(quality["Method"] == m) & (quality["Label"] == c)]["Silhouette"].mean()
                    for c in label_names]
        vals_knn = [quality[(quality["Method"] == m) & (quality["Label"] == c)]["kNNPurity"].mean()
                    for c in label_names]
        ax_sil.bar(x + (i - 1) * w, vals_sil, width=w, color=colors[m], edgecolor="white", label=m)
        ax_knn.bar(x + (i - 1) * w, vals_knn, width=w, color=colors[m], edgecolor="white", label=m)
    ax_sil.set_xticks(x); ax_sil.set_xticklabels([label_display[c] for c in label_names])
    ax_knn.set_xticks(x); ax_knn.set_xticklabels([label_display[c] for c in label_names])
    ax_sil.set_ylabel("Silhouette score", fontsize=11)
    ax_knn.set_ylabel("kNN label purity (k=10)", fontsize=11)
    ax_sil.set_title("(a) Cluster separation", fontsize=12, fontweight="bold")
    ax_knn.set_title("(b) Neighborhood coherence", fontsize=12, fontweight="bold")
    ax_sil.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_knn.set_ylim(0, 1.02)
    for a in (ax_sil, ax_knn):
        a.legend(title="Embedding", fontsize=8, title_fontsize=9, loc="best")
    fig.suptitle("Embedding Structure Quality across Chemical Labels",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig03_structure_quality")


# ═══════════════════════════════════════════════════════════
# Fig 3b — t-SNE Colored by Chemical Properties (Supplementary)
# ═══════════════════════════════════════════════════════════
def fig_tsne_by_chemistry():
    """Supplementary figure: cross-embedding t-SNE comparison colored by
    metal element. One row of three panels (CompGCN / Node2Vec / TransE),
    each annotated with its silhouette score on the metal labels."""
    print("Fig supp: 3-panel t-SNE by metal (CompGCN / Node2Vec / TransE)...")

    chem_path = os.path.join(HPC, "studies", "data", "chemcial_properties.csv")
    if not os.path.exists(chem_path):
        print("  [SKIP] chemical properties not found")
        return
    chem_df = pd.read_csv(chem_path, low_memory=False)
    if "metal_element" not in chem_df.columns and "metal_cluster_elements" in chem_df.columns:
        chem_df["metal_element"] = chem_df["metal_cluster_elements"].apply(_primary_metal)
    if "metal_element" not in chem_df.columns:
        print("  [SKIP] metal_element / metal_cluster_elements columns missing")
        return

    metal_sub = chem_df.dropna(subset=["metal_element"]).copy()
    top_metals = metal_sub["metal_element"].value_counts().head(8).index.tolist()
    metal_sub["metal_cat"] = metal_sub["metal_element"].where(
        metal_sub["metal_element"].isin(top_metals), other="Other"
    )
    cats = top_metals + ["Other"]
    cat_colors = {c: PLASMA(i / (len(cats) - 1)) for i, c in enumerate(cats)}

    methods = ["CompGCN", "Node2Vec", "TransE"]
    quality_path = os.path.join(STUDIES, "concept_vectors", "embedding_structure_quality.csv")
    try:
        qdf = pd.read_csv(quality_path)
    except Exception:
        qdf = pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ok = 0
    for ax, method in zip(axes, methods):
        proj_path = os.path.join(STUDIES, "concept_vectors", f"projection_points_{method}.csv")
        if not os.path.exists(proj_path):
            ax.text(0.5, 0.5, f"No projection for {method}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(method, fontsize=12, fontweight="bold")
            continue
        pdf = pd.read_csv(proj_path)
        merged = pdf.merge(metal_sub[["mof_uri", "metal_cat"]], on="mof_uri", how="inner")
        for cat in cats:
            sub = merged[merged["metal_cat"] == cat]
            if len(sub) == 0:
                continue
            ax.scatter(sub["x"], sub["y"], c=[cat_colors[cat]], s=3, alpha=0.55,
                       label=cat, rasterized=True)
        sil_row = qdf[(qdf["Method"] == method) & (qdf["Label"] == "metal_element")]
        if not sil_row.empty:
            sil = float(sil_row["Silhouette"].values[0])
            knn = float(sil_row["kNNPurity"].values[0])
            n = int(sil_row["N"].values[0])
            ax.text(0.02, 0.98,
                    f"N = {n:,}\nsilhouette = {sil:.2f}\nkNN purity = {knn:.2f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2" if ax is axes[0] else "")
        ax.tick_params(labelbottom=False, labelleft=False)
        ok += 1

    if ok == 0:
        print("  [SKIP] No panels could be rendered")
        plt.close(fig)
        return
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(cats), fontsize=8,
                   markerscale=3, frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("t-SNE Projections of KG Embeddings Colored by Metal Element",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_supp_tsne_chemistry")
    return


# ═══════════════════════════════════════════════════════════
# Fig 4 — Link Prediction AUC Comparison
# ═══════════════════════════════════════════════════════════
def fig_link_prediction_auc():
    print("Fig 4: Link prediction AUC...")
    # Use family-aware results (more rigorous evaluation)
    df = pd.read_csv(os.path.join(STUDIES, "link_prediction_family_eval", "family_aware_link_prediction.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar colors
    colors = [PLASMA(0.3), PLASMA(0.7)]
    x = range(len(df))
    bars = ax.bar(x, df["AUC"], color=colors, edgecolor="white", linewidth=0.5)

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Link Prediction Performance (Family-Aware Evaluation)", fontsize=13, fontweight="bold")

    # Annotate bars with AUC and Hits@10
    for i, (auc, hits10) in enumerate(zip(df["AUC"], df["Hits@10"])):
        ax.text(i, auc + 0.02, f"AUC: {auc:.3f}\nHits@10: {hits10:.3f}",
                ha="center", fontsize=9, fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random baseline")
    ax.legend(fontsize=8)

    save_fig(fig, "fig04_link_prediction_auc")


# ═══════════════════════════════════════════════════════════
# Fig 5 — Relation-Level Link Prediction Quality
# ═══════════════════════════════════════════════════════════
def fig_relation_quality():
    """Fig 5 — Per-relation family-aware LP AUC, grouped by embedding method.
    Family-aware means parent/derivative MOFs co-clustered in either train or
    test so learned edges cannot leak across groups."""
    print("Fig 5: Per-relation family-aware LP quality...")
    summary_path = os.path.join(STUDIES, "link_prediction_family_eval",
                                 "family_aware_relation_quality_summary.csv")
    if not os.path.exists(summary_path):
        print("  [SKIP] family_aware_relation_quality_summary.csv not found")
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        print("  [SKIP] relation-quality summary is empty")
        return
    methods = [m for m in ["CompGCN", "Node2Vec", "TransE"] if m in df["method"].unique().tolist()]
    relations = sorted(df["relation"].unique().tolist())
    colors = {"CompGCN": PLASMA(0.15), "Node2Vec": PLASMA(0.55), "TransE": PLASMA(0.85)}

    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.55 * len(relations) * len(methods)))
    y_positions = []
    y_labels = []
    y = 0
    for rel in relations:
        for m in methods:
            sub = df[(df["method"] == m) & (df["relation"] == rel)]
            if sub.empty:
                continue
            auc = float(sub["AUC_mean"].iloc[0])
            err = float(sub["AUC_std"].iloc[0]) if not sub["AUC_std"].isna().all() else 0.0
            n_edges = int(sub["n_edges_total"].iloc[0])
            ax.barh(y, auc, xerr=err, color=colors.get(m, PLASMA(0.5)),
                    edgecolor="white", linewidth=0.3,
                    error_kw={"linewidth": 1, "ecolor": "black"})
            ax.text(auc + max(err, 0.003) + 0.005, y,
                    f"{auc:.3f} (n={n_edges:,})", va="center", fontsize=7)
            y_positions.append(y)
            y_labels.append(f"{rel} — {m}")
            y += 1
        y += 0.4
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(0.5, 1.05)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Random")
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[m], label=m) for m in methods]
    ax.legend(handles=legend_elements, title="Embedding", fontsize=8,
              title_fontsize=9, loc="lower right")
    ax.set_xlabel("AUC-ROC (family-aware, mean ± SD over 3 seeds)", fontsize=10)
    ax.set_title("Per-Relation Link Prediction (Family-Aware Splits)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig05_relation_quality")


# ═══════════════════════════════════════════════════════════
# Fig 6 — Chemical Property Prediction Heatmap
# ═══════════════════════════════════════════════════════════
def fig_prediction_heatmap():
    print("Fig 6: Prediction heatmap...")
    df = pd.read_csv(os.path.join(STUDIES, "chem_combo_compare", "chem_combo_best_r2.csv"))

    # Create a combined label
    df["Method"] = df["Embedding"] + "\n" + df["FeatureFamily"]

    # Get top 20 targets by max R2
    max_r2 = df.groupby("Target")["R2"].max().sort_values(ascending=False)
    top_targets = max_r2.head(20).index.tolist()
    df_top = df[df["Target"].isin(top_targets)]

    # Pivot
    col_order = ["CompGCN\nHybrid", "CompGCN\nKGOnly", "ChemOnly\nChemOnly",
                 "TransE\nHybrid", "TransE\nKGOnly", "Node2Vec\nHybrid", "Node2Vec\nKGOnly"]
    pivot = df_top.pivot_table(values="R2", index="Target", columns="Method")

    # Reorder columns
    existing_cols = [c for c in col_order if c in pivot.columns]
    pivot = pivot[existing_cols]

    # Sort rows by max R2
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, cmap="plasma", annot=True, fmt=".2f", ax=ax,
                vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "R$^2$", "shrink": 0.8})

    ax.set_title("Property Prediction R$^2$ by Embedding and Feature Family",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=0, fontsize=8)
    plt.yticks(fontsize=8)

    save_fig(fig, "fig06_prediction_heatmap")


# ═══════════════════════════════════════════════════════════
# Fig 7 — Best R² per Target
# ═══════════════════════════════════════════════════════════
def fig_best_r2_per_target():
    """Fig 7 — Best-R² per target, split into two panels so absolute CO2/H2O
    binding-energy targets aren't juxtaposed against easily-learned density/
    pore targets. Absolute DAC binding energies are labeled as such to
    disambiguate from the Δ(BE) targets reported in Fig. 12."""
    print("Fig 7: Best R² per target (two-panel)...")
    df = pd.read_csv(os.path.join(STUDIES, "chem_combo_compare", "best_model_per_target.csv"))

    def _target_group(t):
        tl = t.lower()
        if ("co2" in tl or "h2o" in tl) and ("binding" in tl or "henry" in tl or "widom" in tl):
            return "DAC binding (absolute)"
        return "Structural / DFT / uptake"

    def _relabel(t):
        tl = t.lower()
        if tl == "co2 binding energy":
            return "CO$_2$ binding energy (absolute)"
        if tl == "h2o binding energy":
            return "H$_2$O binding energy (absolute)"
        return t

    df["Group"] = df["Target"].apply(_target_group)
    df["Target_disp"] = df["Target"].apply(_relabel)

    embed_map = {"CompGCN": PLASMA(0.15), "ChemOnly": PLASMA(0.45),
                 "TransE": PLASMA(0.65), "Node2Vec": PLASMA(0.85)}
    struct = df[df["Group"] == "Structural / DFT / uptake"].sort_values("R2", ascending=True)
    dac = df[df["Group"] == "DAC binding (absolute)"].sort_values("R2", ascending=True)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 11), gridspec_kw={"width_ratios": [3, 1]}
    )

    for ax, sub, title in [(ax_left, struct, "(a) Structural / DFT / uptake properties"),
                           (ax_right, dac, "(b) DAC binding (absolute)")]:
        colors = [embed_map.get(e, PLASMA(0.5)) for e in sub["Embedding"]]
        ax.barh(range(len(sub)), sub["R2"], color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["Target_disp"], fontsize=7)
        ax.set_xlabel("Best R$^2$", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1.05)
        for i, v in enumerate(sub["R2"].values):
            ax.text(max(v, 0) + 0.01, i, f"{v:.3f}", va="center", fontsize=6)
        ax.axvline(0, color="gray", linewidth=0.5)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=e) for e, c in embed_map.items()]
    ax_left.legend(handles=legend_elements, title="Best Embedding",
                   loc="lower right", fontsize=8, title_fontsize=9)

    fig.suptitle(
        "Best $R^2$ per Target (Δ CO$_2$/H$_2$O binding upon functionalization are in Fig. 12)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "fig07_best_r2_per_target")


# ═══════════════════════════════════════════════════════════
# Fig 8 — Feature Family Comparison
# ═══════════════════════════════════════════════════════════
def fig_feature_family_comparison():
    print("Fig 8: Feature family comparison...")
    df_full = pd.read_csv(os.path.join(STUDIES, "chem_combo_compare", "chem_combo_best_r2.csv"))

    agg = (df_full.groupby(["Embedding", "FeatureFamily"], as_index=False)
                  .agg(R2_mean=("R2", "mean"), R2_std=("R2", "std")))
    agg["R2_std"] = agg["R2_std"].fillna(0.0)

    methods = ["CompGCN", "Node2Vec", "TransE"]
    families = ["KGOnly", "Hybrid"]
    family_pretty = {"ChemOnly": "chem descriptors", "KGOnly": "KG embedding", "Hybrid": "hybrid"}
    fam_colors = {"ChemOnly": PLASMA(0.15), "KGOnly": PLASMA(0.5), "Hybrid": PLASMA(0.85)}

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(methods))
    w = 0.26

    # ChemOnly baseline (same value across all methods)
    chem_row = agg[agg["FeatureFamily"] == "ChemOnly"]
    if len(chem_row):
        chem_val = float(chem_row["R2_mean"].iloc[0])
        chem_err = float(chem_row["R2_std"].iloc[0])
        ax.bar(x - w, [chem_val] * len(methods), width=w, yerr=[chem_err] * len(methods),
               color=fam_colors["ChemOnly"], edgecolor="white", linewidth=0.3,
               label=family_pretty["ChemOnly"],
               error_kw={"linewidth": 1, "ecolor": "black", "capsize": 2})

    # KGOnly and Hybrid (vary by embedding method)
    for i, fam in enumerate(families):
        vals, errs = [], []
        for m in methods:
            row = agg[(agg["Embedding"] == m) & (agg["FeatureFamily"] == fam)]
            vals.append(float(row["R2_mean"].iloc[0]) if len(row) else np.nan)
            errs.append(float(row["R2_std"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + i * w, vals, width=w, yerr=errs,
               color=fam_colors[fam], edgecolor="white", linewidth=0.3,
               label=family_pretty[fam],
               error_kw={"linewidth": 1, "ecolor": "black", "capsize": 2})

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean Best R$^2$", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Property Prediction: Mean Best R$^2$ by Embedding and Feature Family",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Features", fontsize=8, title_fontsize=9, loc="best")

    save_fig(fig, "fig08_feature_family_comparison")


# ═══════════════════════════════════════════════════════════
# Fig 9 — Imputation F1 Comparison
# ═══════════════════════════════════════════════════════════
def fig_imputation_comparison():
    """Fig 9 — Imputation F1 using KG embeddings for MOFs lacking chemical data."""
    print("Fig 9: Imputation F1 comparison...")
    cv_path = os.path.join(STUDIES, "imputation_compare_cv", "imputation_summary.csv")
    legacy_path = os.path.join(STUDIES, "imputation_compare", "imputation_summary.csv")
    summary_path = cv_path if os.path.exists(cv_path) else legacy_path
    df = pd.read_csv(summary_path)
    has_std = "WeightedF1_std" in df.columns

    family_map = {"kg": "KG embedding", "hybrid": "hybrid"}
    family_order = ["kg", "hybrid"]
    colors = {"KG embedding": PLASMA(0.3), "hybrid": PLASMA(0.75)}
    methods = ["CompGCN", "Node2Vec", "TransE"]

    targets = df["Target"].unique().tolist()
    fig, axes = plt.subplots(1, len(targets), figsize=(7.2 * len(targets), 5.5))
    if len(targets) == 1:
        axes = [axes]
    target_titles = {"topology": "Topology classification",
                     "metal_element": "Metal-element classification"}

    for ax, target in zip(axes, targets):
        sub = df[df["Target"] == target].copy()
        width = 0.35
        x = np.arange(len(methods))
        for i, fam in enumerate(family_order):
            fam_pretty = family_map.get(fam, fam)
            vals = []
            errs = []
            for m in methods:
                r = sub[(sub["FeatureFamily"] == fam) & (sub["Embedding"] == m)]
                vals.append(float(r["WeightedF1"].iloc[0]) if len(r) else 0.0)
                errs.append(float(r["WeightedF1_std"].iloc[0]) if (len(r) and has_std) else 0.0)
            ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs,
                   color=colors[fam_pretty], edgecolor="white", linewidth=0.3,
                   label=fam_pretty,
                   error_kw={"linewidth": 1, "ecolor": "black", "capsize": 2})
        base = sub["BaselineAccuracy"].dropna()
        if not base.empty:
            b = float(base.mean())
            ax.axhline(b, color="red", linestyle="--", linewidth=1.2,
                       label=f"Majority baseline ({b:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Weighted F1", fontsize=11)
        ax.set_title(target_titles.get(target, target), fontsize=12, fontweight="bold")
        ax.legend(title="Features", fontsize=8, title_fontsize=9, loc="lower right")

    fig.suptitle("Label Imputation from KG Embeddings",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig09_imputation_comparison")


# ═══════════════════════════════════════════════════════════
# Fig 10 — Concept Similarity Heatmap
# ═══════════════════════════════════════════════════════════
def fig_concept_similarity():
    print("Fig 10: Concept similarity...")
    df = pd.read_csv(os.path.join(STUDIES, "concept_vectors", "concept_similarity_CompGCN.csv"),
                     index_col=0)

    # Filter out linker SMILES columns/rows
    keep = [c for c in df.columns if "linker_smiles" not in c]
    df = df.loc[keep, keep]

    # Omit metal-cluster element concepts and malformed topology label
    keep = [
        c for c in df.columns
        if not str(c).startswith("metal_cluster_elements__")
        and str(c) != "topology__is__ERROR_UNKNOWN"
    ]
    df = df.loc[keep, keep]

    # Clean labels
    def clean_label(s):
        s = str(s)
        s = s.replace("__is__", ": ").replace("__has__", ": ")
        s = s.replace("_", " ").replace("  ", " ").strip()
        if len(s) > 30:
            s = s[:27] + "..."
        return s

    df.index = [clean_label(i) for i in df.index]
    df.columns = [clean_label(c) for c in df.columns]

    # Colorbar on the right keeps long row labels clear on the left.
    # Font sizes geared for print (single-column or moderate reduction in layout).
    fig, ax = plt.subplots(figsize=(16, 11.5), layout="constrained")
    im = ax.imshow(df.values, cmap="plasma", aspect="auto", vmin=-1, vmax=1)

    pub_tick = 16
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, fontsize=pub_tick, rotation=90)
    ax.set_yticklabels(df.index, fontsize=pub_tick)
    ax.tick_params(axis="both", labelsize=pub_tick)

    cbar = fig.colorbar(im, ax=ax, location="right", pad=0.03, shrink=0.82)
    cbar.set_label("Pearson Correlation", fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    ax.set_title("Concept Vector Similarity (CompGCN)", fontsize=26, fontweight="bold", pad=18)

    fig.savefig(os.path.join(OUT, "fig10_concept_similarity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig10_concept_similarity.png")


# ═══════════════════════════════════════════════════════════
# Fig 11 — Concept Probe Performance
# ═══════════════════════════════════════════════════════════
def fig_concept_probe_performance():
    """Fig 11 — Concept probe ROC-AUC with mean ± SD over multiple seeds.
    Source CSV emits one row per (method, concept, split_seed); we aggregate."""
    print("Fig 11: Concept probe performance (multi-seed)...")
    df = pd.read_csv(os.path.join(STUDIES, "concept_vectors", "concept_probe_metrics.csv"))

    agg = (df.groupby(["method", "concept"], as_index=False)
             .agg(roc_auc_mean=("roc_auc", "mean"),
                  roc_auc_std=("roc_auc", "std"),
                  n_seeds=("split_seed", "nunique")))
    agg["roc_auc_std"] = agg["roc_auc_std"].fillna(0.0)

    # Get top 20 first, then filter out unwanted categories
    compgcn = agg[agg["method"] == "CompGCN"].nlargest(20, "roc_auc_mean")
    
    # Filter out specific unwanted categories from the top 20
    exclude_patterns = [
        "topology__is__ERROR_UNKNOWN",
        "metal_cluster_elements__has__O",
        "linker_smiles__has___N_N_N_"
    ]
    compgcn = compgcn[~compgcn["concept"].isin(exclude_patterns)]
    top_concepts = compgcn["concept"].tolist()

    def clean_concept(s):
        s = str(s).replace("__is__", ": ").replace("__has__", ": ")
        s = s.replace("_", " ").replace("  ", " ").strip()
        return s

    methods = [m for m in ["CompGCN", "Node2Vec", "TransE"] if m in agg["method"].unique().tolist()]
    colors = {"CompGCN": PLASMA(0.15), "Node2Vec": PLASMA(0.55), "TransE": PLASMA(0.85)}

    x = np.arange(len(top_concepts))
    n_methods = len(methods)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, m in enumerate(methods):
        means, stds = [], []
        for c in top_concepts:
            row = agg[(agg["method"] == m) & (agg["concept"] == c)]
            means.append(float(row["roc_auc_mean"].iloc[0]) if len(row) else np.nan)
            stds.append(float(row["roc_auc_std"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (i - (n_methods - 1) / 2) * width, means, width,
               yerr=stds, color=colors[m], edgecolor="white", linewidth=0.3,
               error_kw={"linewidth": 0.8, "ecolor": "black", "capsize": 2},
               label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([clean_concept(c) for c in top_concepts],
                       rotation=60, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylabel("Linear-probe ROC-AUC", fontsize=13, fontweight="bold")
    ax.tick_params(axis='y', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    n_seeds = int(agg["n_seeds"].max()) if "n_seeds" in agg.columns else 1
    ax.set_title("Concept Probe Performance (Top Concepts)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.02)
    ax.legend(title="Embedding", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig11_concept_probe_performance")


# ═══════════════════════════════════════════════════════════
# Fig 12 — Functionalization Delta Prediction
# ═══════════════════════════════════════════════════════════
def fig_relation_prediction():
    """Fig 12 — Δ CO₂ / Δ H₂O prediction with 5-fold CV error bars.
    Grouped by embedding, colored by feature family (KG, chem, hybrid).
    Δ is defined as child_BE − parent_BE over syn:derivedFrom pairs."""
    print("Fig 12: Functionalization Δ prediction (CV error bars)...")

    cv_metrics_path = os.path.join(STUDIES, "relation_compare_cv",
                                    "relation_metrics_comparison.csv")
    legacy_metrics_path = os.path.join(STUDIES, "relation_compare",
                                        "relation_metrics_comparison.csv")
    if os.path.exists(cv_metrics_path):
        raw = pd.read_csv(cv_metrics_path)
    elif os.path.exists(legacy_metrics_path):
        raw = pd.read_csv(legacy_metrics_path)
    else:
        print("  [SKIP] relation metrics CSV not found")
        return

    if "AmineType" in raw.columns:
        raw = raw[raw["AmineType"] == "ALL"]
    agg = (raw.groupby(["Embedding", "FeatureFamily", "Target", "Model"], as_index=False)
              .agg(R2_mean=("R2", "mean"),
                   R2_std=("R2", "std"),
                   n_folds=("Fold", "nunique") if "Fold" in raw.columns else ("R2", "count")))
    agg["R2_std"] = agg["R2_std"].fillna(0.0)
    best_rows = agg.loc[agg.groupby(["Embedding", "FeatureFamily", "Target"])["R2_mean"].idxmax()]

    methods = ["CompGCN", "Node2Vec", "TransE"]
    families = ["chem", "kg", "hybrid"]
    family_pretty = {"chem": "chem descriptors", "kg": "KG embedding", "hybrid": "hybrid"}
    fam_colors = {"chem": PLASMA(0.15), "kg": PLASMA(0.5), "hybrid": PLASMA(0.85)}

    targets_ordered = [t for t in ["Delta_CO2", "Delta_H2O"] if t in best_rows["Target"].unique().tolist()]
    fig, axes = plt.subplots(1, max(len(targets_ordered), 1),
                              figsize=(7.2 * max(len(targets_ordered), 1), 5.2))
    if len(targets_ordered) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets_ordered):
        sub = best_rows[best_rows["Target"] == target]
        x = np.arange(len(methods))
        w = 0.26
        for i, fam in enumerate(families):
            vals, errs = [], []
            for m in methods:
                row = sub[(sub["Embedding"] == m) & (sub["FeatureFamily"] == fam)]
                vals.append(float(row["R2_mean"].iloc[0]) if len(row) else np.nan)
                errs.append(float(row["R2_std"].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (i - 1) * w, vals, width=w, yerr=errs,
                   color=fam_colors[fam], edgecolor="white", linewidth=0.3,
                   label=family_pretty[fam],
                   error_kw={"linewidth": 1, "ecolor": "black", "capsize": 2})
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        title = "(a) ΔCO$_2$ binding energy" if target == "Delta_CO2" else "(b) ΔH$_2$O binding energy"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("R$^2$ (5-fold CV)", fontsize=11)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.legend(title="Features", fontsize=8, title_fontsize=9, loc="best")

    fig.suptitle(
        "Predicting the Amine-Functionalization Effect on CO$_2$/H$_2$O Binding Energies",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "fig12_functionalization_prediction")


# ═══════════════════════════════════════════════════════════
# Fig 13 — DAC Screening Score Distribution
# ═══════════════════════════════════════════════════════════
def fig_dac_screening():
    print("Fig 13: DAC screening...")
    df = pd.read_csv(os.path.join(STUDIES, "dac_screen", "real_mof_ranked.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

    # Panel (a): DAC score distribution
    scores = df["dac_score"].dropna()
    sns.histplot(scores, bins=50, ax=ax1, color=PLASMA(0.5), edgecolor="white", linewidth=0.3)
    ax1.axvline(x=0.8377, color=PLASMA(0.95), linestyle="--", linewidth=2,
                label="Top candidate (0.838)")
    ax1.set_xlabel("DAC Score", fontsize=16)
    ax1.set_ylabel("Count", fontsize=16)
    ax1.set_title("(a) DAC Score Distribution", fontsize=18, fontweight="bold")
    ax1.legend(fontsize=13)
    ax1.tick_params(labelsize=14)

    # Panel (b): CO2 vs Hydrophobic score, colored by stability
    mask = df["co2_score"].notna() & df["hydrophobic_score"].notna() & df["stability_score"].notna()
    sub = df[mask]

    scatter = ax2.scatter(sub["co2_score"], sub["hydrophobic_score"],
                          c=sub["stability_score"], cmap="plasma",
                          s=8, alpha=0.5, rasterized=True)

    # Highlight top candidate
    top = df.iloc[0]
    if pd.notna(top["co2_score"]) and pd.notna(top["hydrophobic_score"]):
        ax2.scatter([top["co2_score"]], [top["hydrophobic_score"]],
                    c="red", s=900, marker="*", zorder=10, edgecolors="white",
                    linewidths=2.0, label="Top candidate")
        ax2.legend(fontsize=13)

    cbar = fig.colorbar(scatter, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label("Stability Score", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax2.set_xlabel("CO$_2$ Affinity Score", fontsize=16)
    ax2.set_ylabel("Hydrophobicity Score", fontsize=16)
    ax2.set_title("(b) DAC Property Space", fontsize=18, fontweight="bold")
    ax2.tick_params(labelsize=14)

    fig.suptitle("Direct Air Capture Screening Results", fontsize=20, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig13_dac_screening")


# ═══════════════════════════════════════════════════════════
# Fig 14 — Semantic Search Results
# ═══════════════════════════════════════════════════════════
def fig_semantic_search():
    """Fig 14 (supplement) — Slim count panel summarizing aggregate SPARQL
    queries over the KG. DAC-specific chemist queries and their top rows are
    reported in Table 7 rather than as a figure."""
    print("Fig 14: Semantic-search count panel (supplement)...")
    import os as _os
    ssd = _os.path.join(STUDIES, "semantic_search_demo")

    def _safe_read(fname):
        p = _os.path.join(ssd, fname)
        return pd.read_csv(p) if _os.path.exists(p) else pd.DataFrame()

    mof_types = _safe_read("count_mofs_by_type_results.csv")
    metals = _safe_read("metal_element_distribution_results.csv")
    topo = _safe_read("mofs_with_topology_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if not mof_types.empty:
        colors = [PLASMA(i / max(len(mof_types) - 1, 1)) for i in range(len(mof_types))]
        axes[0].barh(mof_types["type"], mof_types["count"], color=colors, edgecolor="white")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Count (log scale)", fontsize=10)
        axes[0].set_title("(a) MOF types", fontsize=12, fontweight="bold")
        for y, cnt in enumerate(mof_types["count"]):
            axes[0].text(cnt * 1.05, y, f"{int(cnt):,}", va="center", fontsize=8)
    if not metals.empty:
        metals_top = metals.head(12)
        colors_m = [PLASMA(i / max(len(metals_top) - 1, 1)) for i in range(len(metals_top))]
        axes[1].barh(metals_top["element"], metals_top["count"], color=colors_m, edgecolor="white")
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Number of MOFs", fontsize=10)
        axes[1].set_title("(b) Metal elements (top 12)", fontsize=12, fontweight="bold")
    if not topo.empty:
        topo_top = topo.head(12)
        colors_t = [PLASMA(i / max(len(topo_top) - 1, 1)) for i in range(len(topo_top))]
        axes[2].barh(topo_top["topo"], topo_top["count"], color=colors_t, edgecolor="white")
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Number of MOFs", fontsize=10)
        axes[2].set_title("(c) Topologies (top 12)", fontsize=12, fontweight="bold")

    fig.suptitle("Aggregate KG Queries (DAC-specific chemist queries appear in Table 7)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig14_semantic_search")


# ═══════════════════════════════════════════════════════════
# Fig 15 — Novel MOF Composition Validation
# ═══════════════════════════════════════════════════════════
def fig_composition_validation():
    print("Fig 15: Embedding composition validation...")
    import os as _os
    val_df = pd.read_csv(_os.path.join(STUDIES, "dac_screen", "composition_validation.csv"))
    pred_df = pd.read_csv(_os.path.join(STUDIES, "dac_screen", "novel_mof_embedding_predictions.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Cosine similarity distribution with error bars via KDE
    ax = axes[0]
    cos_sims = val_df["cosine_similarity"].values
    n_bins = 30
    counts, bin_edges = np.histogram(cos_sims, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9,
           color=PLASMA(0.6), edgecolor="white", linewidth=0.3)

    mean_val = cos_sims.mean()
    std_val = cos_sims.std()
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_val:.3f}")
    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.15, color="gray",
               label=f"$\\pm$1 SD ({std_val:.3f})")
    ax.axvline(0.7, color="red", linestyle=":", linewidth=1.5,
               label=f"Target > 0.7 ({(cos_sims > 0.7).mean() * 100:.1f}\\%)")

    ax.set_xlabel("Cosine Similarity (composed vs. actual)", fontsize=10)
    ax.set_ylabel("Number of MOFs", fontsize=10)
    ax.set_title(f"(a) Composition Fidelity (N = {len(val_df)})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # Panel B: Predicted DAC score distribution for novel candidates
    ax = axes[1]
    dac_scores = pred_df["predicted_dac_score"].values
    ax.hist(dac_scores, bins=40, color=PLASMA(0.3), edgecolor="white", linewidth=0.3)
    top_k = 10
    threshold = np.sort(dac_scores)[-top_k]
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Top-{top_k} threshold")
    ax.set_xlabel("Predicted DAC Score (density concept alignment)", fontsize=10)
    ax.set_ylabel("Number of candidates", fontsize=10)
    ax.set_title(f"(b) Novel Candidates Scored (N = {len(pred_df):,})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Embedding Composition for Novel MOF Prediction",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig15_composition_validation")


# ═══════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════
def _latex_escape(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}")
                  .replace("_", "\\_")
                  .replace("#", "\\#")
                  .replace("&", "\\&")
                  .replace("%", "\\%")
                  .replace("{", "\\{")
                  .replace("}", "\\}")
                  .replace("$", "\\$"))


def _short_id(uri: str) -> str:
    u = str(uri)
    return u.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _build_dac_semantic_search_table() -> str:
    """Assemble Table 7 of DAC-chemist SPARQL queries with top rows from the
    semantic_search_demo results directory. Degrades gracefully if CSVs missing."""
    ssd = os.path.join(STUDIES, "semantic_search_demo")

    def _read(fname):
        p = os.path.join(ssd, fname)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    queries = [
        {
            "question": "List experimental MOFs with large pores and low density (DAC-capacity candidates).",
            "pattern": r"PLD $>6$\,\AA{} $\wedge$ Density $<1.0$\,g/cm\textsuperscript{3}",
            "csv": "dac_porous_low_density_results.csv",
        },
        {
            "question": "Find strong CO\\textsubscript{2} binders with weaker H\\textsubscript{2}O affinity.",
            "pattern": r"CO\textsubscript{2} BE $<-0.5$\,eV $\wedge$ CO\textsubscript{2} BE $<$ H\textsubscript{2}O BE",
            "csv": "dac_strong_co2_binders_results.csv",
        },
        {
            "question": "Return Mg-containing MOFs with pcu/fcu topology (Mg-MOF-74 analogues).",
            "pattern": r"hasMetalElement $=$ Mg $\wedge$ topologyCode $\in \{pcu, fcu\}$",
            "csv": "dac_mg_family_analogues_results.csv",
        },
        {
            "question": "Amine derivatives whose functionalization improves CO\\textsubscript{2} binding vs. their parent.",
            "pattern": r"syn:derivedFrom $\wedge$ child CO\textsubscript{2} BE $<$ parent CO\textsubscript{2} BE",
            "csv": "dac_functionalization_improving_co2_results.csv",
        },
    ]

    rows = []
    for q in queries:
        df_q = _read(q["csv"])
        n = len(df_q)
        if n == 0:
            sample = "(no results; query not yet run or no matches)"
        else:
            # Pick first URI-like column + one salient numeric column.
            uri_col = next((c for c in df_q.columns if df_q[c].dtype == object), None)
            num_cols = [c for c in df_q.columns if c != uri_col and pd.api.types.is_numeric_dtype(df_q[c])]
            examples = []
            for _, r in df_q.head(3).iterrows():
                name = _latex_escape(_short_id(r[uri_col])) if uri_col else ""
                extras = []
                for nc in num_cols[:2]:
                    v = r[nc]
                    if pd.notna(v):
                        extras.append(f"{_latex_escape(nc)}={float(v):.2f}")
                line = name
                if extras:
                    line += " (" + ", ".join(extras) + ")"
                examples.append(line)
            sample = f"{n:,} matches. Examples: " + "; ".join(examples)
        rows.append((q["question"], q["pattern"], sample))

    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{DAC-chemist SPARQL queries over the MOFology KG "
        "(8.4M triples). Top-3 rows shown where available; full result sets "
        "are written to \\texttt{results/semantic\\_search\\_demo/}.}\n"
        "\\label{tab:semantic_search}\n\\small\n"
        "\\begin{tabular}{p{4.2cm}p{3.6cm}p{7.2cm}}\n"
        "\\toprule\n"
        "\\textbf{Chemist Question} & \\textbf{Query Pattern} & \\textbf{Result summary} \\\\\n"
        "\\midrule\n"
    )
    for q, p, s in rows:
        tex += f"{q} & {p} & {s} \\\\\n\\addlinespace\n"
    tex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    return tex


def generate_tables():
    print("\nGenerating LaTeX tables...")

    # Table 2: KG Statistics (from actual semantic search results)
    kg_stats = r"""\begin{table}[htbp]
\centering
\caption{MOFology Knowledge Graph Statistics}
\label{tab:kg_stats}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Count} \\
\midrule
Total MOFs & 254,099 \\
\quad Experimental (CSD) & 40,791 \\
\quad Hypothetical & 213,308 \\
\quad Functionalized (amine-grafted) & 2,650 \\
Total Triples & 8,399,032 \\
Unique Organic Linkers & $\sim$5,000 \\
Unique Metal Clusters & $\sim$1,400 \\
Unique Topologies & $\sim$1,600 \\
Distinct Properties & 95 \\
Properties ($\geq$500 samples) & 45 \\
Relation Types & 15 \\
Data Sources & 6 \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLES, "table02_kg_stats.tex"), "w") as f:
        f.write(kg_stats)
    print("  [OK] table02_kg_stats.tex")

    # Table 3: Embedding method comparison (family-aware LP mean ± SD + chem_combo)
    summary_path = os.path.join(STUDIES, "link_prediction_family_eval",
                                 "family_aware_link_prediction_summary.csv")
    if os.path.exists(summary_path):
        lp_df = pd.read_csv(summary_path)
        lp_has_std = True
    else:
        lp_df = pd.read_csv(os.path.join(STUDIES, "link_prediction_family_eval",
                                         "family_aware_link_prediction.csv"))
        lp_has_std = False
    cc_df = pd.read_csv(os.path.join(STUDIES, "chem_combo_compare",
                                     "chem_combo_summary.csv"))

    tex = r"""\begin{table}[htbp]
\centering
\caption{KG Embedding Benchmark. Link prediction reports mean $\pm$ SD over 3 seeds on family-aware holdouts that prevent parent--derivative leakage. Property prediction reports mean best $R^2 \pm$ SD across 30 targets.}
\label{tab:embedding_summary}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{LP AUC} & \textbf{LP Hits@10} & \textbf{KG-only $R^2$} & \textbf{Hybrid $R^2$} \\
\midrule
"""
    for method in ["CompGCN", "Node2Vec", "TransE"]:
        lp_row = lp_df[lp_df["method"] == method]
        if len(lp_row) == 0:
            continue
        if lp_has_std:
            auc_mean = float(lp_row["AUC_mean"].values[0])
            auc_std = float(lp_row["AUC_std"].values[0])
            hits_mean = float(lp_row["Hits@10_mean"].values[0])
            hits_std = float(lp_row["Hits@10_std"].values[0])
            auc_s = f"${auc_mean:.3f} \\pm {auc_std:.3f}$"
            hits_s = f"${hits_mean:.3f} \\pm {hits_std:.3f}$"
        else:
            auc_s = f"{float(lp_row['AUC'].values[0]):.3f}"
            hits_s = f"{float(lp_row['Hits@10'].values[0]):.3f}"
        kg_only = cc_df[(cc_df["Embedding"] == method) & (cc_df["FeatureFamily"] == "KGOnly")]
        hybrid = cc_df[(cc_df["Embedding"] == method) & (cc_df["FeatureFamily"] == "Hybrid")]
        if len(kg_only) > 0 and len(hybrid) > 0:
            k_mean = float(kg_only["MeanBestR2"].values[0])
            k_std = float(kg_only["StdBestR2"].values[0])
            h_mean = float(hybrid["MeanBestR2"].values[0])
            h_std = float(hybrid["StdBestR2"].values[0])
            r2_k = f"${k_mean:.3f} \\pm {k_std:.3f}$"
            r2_h = f"${h_mean:.3f} \\pm {h_std:.3f}$"
        else:
            r2_k = "---"
            r2_h = "---"
        tex += f"{method} & {auc_s} & {hits_s} & {r2_k} & {r2_h} \\\\\n"
    # Chem-only baseline
    chem_only = cc_df[cc_df["FeatureFamily"] == "ChemOnly"]
    if len(chem_only) > 0:
        c_mean = float(chem_only["MeanBestR2"].values[0])
        c_std = float(chem_only["StdBestR2"].values[0])
        tex += f"ChemOnly (baseline) & --- & --- & --- & ${c_mean:.3f} \\pm {c_std:.3f}$ \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLES, "table03_embedding_summary.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table03_embedding_summary.tex")

    # Table 4: Top 10 Best Predicted Properties
    best = pd.read_csv(os.path.join(STUDIES, "chem_combo_compare", "best_model_per_target.csv"))
    top10 = best.nlargest(10, "R2")

    tex = r"""\begin{table}[htbp]
\centering
\caption{Top 10 Best Predicted MOF Properties}
\label{tab:top10_properties}
\begin{tabular}{llllc}
\toprule
\textbf{Property} & \textbf{Embedding} & \textbf{Features} & \textbf{Model} & \textbf{R$^2$} \\
\midrule
"""
    for _, row in top10.iterrows():
        target = row["Target"].replace("_", " ")
        tex += f"{target} & {row['Embedding']} & {row['FeatureFamily']} & {row['Model']} & {row['R2']:.4f} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLES, "table04_top10_properties.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table04_top10_properties.tex")

    # Table 5: DAC Candidates
    dac = pd.read_csv(os.path.join(STUDIES, "dac_screen", "real_mof_ranked.csv"), nrows=5)
    tex = r"""\begin{table}[htbp]
\centering
\caption{Top DAC Candidate MOFs}
\label{tab:dac_candidates}
\begin{tabular}{lccccc}
\toprule
\textbf{MOF} & \textbf{Topology} & \textbf{CO$_2$ Score} & \textbf{Hydrophobic} & \textbf{Stability} & \textbf{DAC Score} \\
\midrule
"""
    for _, row in dac.iterrows():
        name = str(row["mof_uri"]).split("#")[-1] if "#" in str(row["mof_uri"]) else str(row["mof_uri"])[-20:]
        topo = row["topology"] if pd.notna(row["topology"]) else "—"
        co2 = f"{row['co2_score']:.3f}" if pd.notna(row["co2_score"]) else "—"
        hydro = f"{row['hydrophobic_score']:.3f}" if pd.notna(row["hydrophobic_score"]) else "—"
        stab = f"{row['stability_score']:.3f}" if pd.notna(row["stability_score"]) else "—"
        dac_s = f"{row['dac_score']:.3f}" if pd.notna(row["dac_score"]) else "—"
        tex += f"{name} & {topo} & {co2} & {hydro} & {stab} & {dac_s} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLES, "table05_dac_candidates.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table05_dac_candidates.tex")

    # Table 6: Property Reliability Weights
    weights = pd.read_csv(os.path.join(STUDIES, "dac_screen", "property_reliability_weights.csv"))
    weights = weights.sort_values("reliability_weight", ascending=False)

    tex = r"""\begin{table}[htbp]
\centering
\caption{Property Prediction Reliability Weights for DAC Screening}
\label{tab:reliability_weights}
\begin{tabular}{lc}
\toprule
\textbf{Property} & \textbf{Reliability Weight} \\
\midrule
"""
    for _, row in weights.iterrows():
        tex += f"{row['property']} & {row['reliability_weight']:.4f} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLES, "table06_reliability_weights.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table06_reliability_weights.tex")

    # Table 7: DAC-relevant semantic search queries with SPARQL patterns and sample rows
    tex = _build_dac_semantic_search_table()
    with open(os.path.join(TABLES, "table07_semantic_search.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table07_semantic_search.tex")

    # Table 8: Top novel MOF candidates from embedding composition
    nov = pd.read_csv(os.path.join(STUDIES, "dac_screen", "novel_mof_embedding_predictions.csv"))
    top_nov = nov.head(10)
    tex = r"""\begin{table}[htbp]
\centering
\caption{Top 10 novel MOF candidates ranked by predicted DAC score. Candidates are previously unobserved metal--linker combinations scored via concept alignment of composed embeddings (mean cosine similarity to actual embeddings: 0.91).}
\label{tab:novel_candidates}
\small
\begin{tabular}{llcc}
\toprule
\textbf{Metal(s)} & \textbf{Linker SMILES (truncated)} & \textbf{DAC Score} & \textbf{Low-Density} \\
\midrule
"""
    for _, row in top_nov.iterrows():
        metal = str(row["metal"])[:12]
        smi = str(row["linker_smiles"])[:40].replace("_", "\\_").replace("#", "\\#")
        dac = row["predicted_dac_score"]
        low_d = row["low_Density"]
        tex += f"{metal} & \\texttt{{{smi}}} & {dac:.3f} & {low_d:.3f} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLES, "table08_novel_candidates.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table08_novel_candidates.tex")

    # Table 9: Composition validation summary
    val = pd.read_csv(os.path.join(STUDIES, "dac_screen", "composition_validation.csv"))
    cos = val["cosine_similarity"]
    tex = r"""\begin{table}[htbp]
\centering
\caption{Embedding composition validation: cosine similarity between composed (metal pseudo-emb + linker pseudo-emb)/2 and actual MOF embedding, over 200 held-out MOFs.}
\label{tab:composition_validation}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""
    tex += f"MOFs validated & {len(val)} \\\\\n"
    tex += f"Mean cosine similarity & {cos.mean():.4f} \\\\\n"
    tex += f"Std.\\ cosine similarity & {cos.std():.4f} \\\\\n"
    tex += f"Minimum & {cos.min():.4f} \\\\\n"
    tex += f"Maximum & {cos.max():.4f} \\\\\n"
    tex += f"\\% with cos $>$ 0.7 & {(cos > 0.7).mean() * 100:.1f}\\% \\\\\n"
    tex += f"\\% with cos $>$ 0.5 & {(cos > 0.5).mean() * 100:.1f}\\% \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(TABLES, "table09_composition_validation.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table09_composition_validation.tex")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("MOFology Paper — Figure Generation")
    print("=" * 60)
    print(f"Output: {OUT}\n")

    fig_ontology_hierarchy()
    fig_property_coverage()
    fig_tsne_embeddings()           # Fig 3 (now structure-quality, not t-SNE)
    fig_tsne_by_chemistry()         # Supplementary 3-panel t-SNE by metal
    # fig_link_prediction_auc() removed per review: info now in table 03.
    fig_relation_quality()
    fig_prediction_heatmap()
    fig_best_r2_per_target()
    fig_feature_family_comparison()
    fig_imputation_comparison()
    fig_concept_similarity()
    fig_concept_probe_performance()
    fig_relation_prediction()
    fig_dac_screening()
    fig_semantic_search()
    fig_composition_validation()
    generate_tables()

    print("\n" + "=" * 60)
    print("All figures and tables generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
