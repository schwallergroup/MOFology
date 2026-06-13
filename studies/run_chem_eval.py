#!/usr/bin/env python3
"""
run_chem_eval.py
================
Focused chemical-property prediction evaluation for CompGCN, TransE, and Node2Vec.

Key improvements vs. run_full_study.py Phase 4:
  - Loads embeddings directly from saved CSVs / .pt files (no GNN forward re-run)
  - L2-normalises all embeddings before regression (unit-sphere)
  - Extracts Free Energy (atom) directly from the KG TTL and adds it as a target
  - Lowers min_samples threshold to 50 to include sparse targets
  - Extended MLP (wider, patience-based early stopping) for harder targets
  - Produces per-method heatmaps, a cross-method R² comparison bar chart,
    actual-vs-predicted scatter plots for the top 12 properties, and a
    comprehensive CSV results table

Usage:
    python studies/run_chem_eval.py
"""

import os
import sys
import re
import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ── logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
KG_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
CHEM_PATH    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv")
COMPGCN_CSV  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_compgcn_embeddings_256d_3layers.csv)
TRANSE_CSV   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings"/mof_transe_embeddings_256d.csv)
N2V_PT       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_256d_p1.0_q1.0.pt)
OUT_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/chem_eval)

SEED         = 42
MIN_SAMPLES  = 50     # minimum labelled MOFs to attempt regression
TEST_FRAC    = 0.20

METADATA_COLS = {
    "mof_uri", "csd_code", "chemical_formula", "mofid",
    "topology", "metal_cluster_elements", "linker_smiles",
    "space_group", "crystal_system",
}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: EXTRACT FREE ENERGY (atom) FROM KG
# ═══════════════════════════════════════════════════════════════════════

def extract_free_energy_from_kg(kg_path: str) -> pd.DataFrame:
    """
    Parse the TTL file and return a DataFrame with columns:
        mof_uri  |  Free Energy (atom)

    Each FreeEnergy node looks like:
        :FreeEnergy_XYZABC_hash a :FreeEnergy, :MaterialProperty ;
            :hasPropertyOwner :MOF_XYZABC ;
            :propertyName "Free Energy (atom)"^^xsd:string ;
            :propertyUnits "eV/atom"^^xsd:string ;
            :propertyValue <number> .
    """
    log.info("  Extracting 'Free Energy (atom)' from KG …")
    BASE = "http://emmo.info/domain-mof/mof-ontology#"

    records: Dict[str, float] = {}   # full_mof_uri -> value
    current_owner = None
    is_free_energy = False

    with open(kg_path) as f:
        for line in f:
            line = line.strip()

            # Detect hasPropertyOwner (sets current MOF)
            m = re.search(r":hasPropertyOwner\s+:MOF_(\S+)", line)
            if m:
                current_owner = BASE + "MOF_" + m.group(1).rstrip(";.").rstrip()
                is_free_energy = False
                continue

            # Detect property name
            if ':propertyName' in line and 'Free Energy (atom)' in line:
                is_free_energy = True
                continue

            # Detect value
            if is_free_energy and current_owner and ':propertyValue' in line:
                vm = re.search(r":propertyValue\s+([\d\.\-eE]+)", line)
                if vm:
                    val = float(vm.group(1))
                    # Keep only the first (or any single) value per MOF
                    # — multiple FreeEnergy nodes per MOF may exist; take mean
                    if current_owner not in records:
                        records[current_owner] = []
                    records[current_owner].append(val)
                is_free_energy = False
                current_owner = None

    # Average multiple values per MOF
    fe_data = {uri: float(np.mean(vals)) for uri, vals in records.items()}
    df_fe = pd.DataFrame(
        [(uri, val) for uri, val in fe_data.items()],
        columns=["mof_uri", "Free Energy (atom) [eV/atom]"],
    )
    log.info("    Extracted Free Energy for %d MOFs", len(df_fe))
    return df_fe


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: LOAD AND NORMALISE EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════

def l2_normalise(df: pd.DataFrame, emb_cols: List[str]) -> pd.DataFrame:
    """In-place L2-normalise the embedding vectors (unit sphere)."""
    X = df[emb_cols].values.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    df = df.copy()
    df[emb_cols] = X / norms
    return df


def load_compgcn(csv_path: str) -> pd.DataFrame:
    log.info("  Loading CompGCN embeddings from %s …", csv_path)
    df = pd.read_csv(csv_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df = l2_normalise(df, emb_cols)
    log.info("    %d MOFs, %d dims  (L2-normalised)", len(df), len(emb_cols))
    return df


def load_transe(csv_path: str) -> pd.DataFrame:
    log.info("  Loading TransE embeddings from %s …", csv_path)
    df = pd.read_csv(csv_path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    # TransE embeddings are already L2-normalised (norms == 1.0)
    norms = np.linalg.norm(df[emb_cols].values, axis=1)
    log.info("    %d MOFs, %d dims  (norms: mean=%.4f — already unit)", 
             len(df), len(emb_cols), norms.mean())
    return df


def load_node2vec_256d(pt_path: str) -> pd.DataFrame:
    """Load all-entity Node2Vec .pt file, filter to MOFs only, return DataFrame."""
    log.info("  Loading Node2Vec 256d embeddings from %s …", pt_path)
    saved    = torch.load(pt_path, weights_only=False)
    all_emb  = saved["embeddings"]        # [N, 256]
    ent2id   = saved["ent2id"]            # uri -> row-index

    rows = []
    for uri, idx in ent2id.items():
        frag = uri.split("#")[-1] if "#" in uri else ""
        if frag.startswith("MOF_") or frag.startswith("FuncMOF_"):
            vec = all_emb[idx].numpy()
            row = {"mof_uri": uri}
            row.update({f"emb_{i}": float(v) for i, v in enumerate(vec)})
            rows.append(row)

    df = pd.DataFrame(rows)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df = l2_normalise(df, emb_cols)
    log.info("    %d MOFs, %d dims  (L2-normalised)", len(df), len(emb_cols))
    return df


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: REGRESSION EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def make_regressors():
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.05,
            n_jobs=-1, random_state=SEED, verbosity=0),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(512, 256, 128),
            max_iter=1000, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
            random_state=SEED),
    }


def evaluate_embedding(
    name: str,
    df_emb: pd.DataFrame,
    df_prop: pd.DataFrame,
) -> List[dict]:
    """
    Merge embeddings with property data, run 4 regressors on every target
    with >= MIN_SAMPLES labelled examples, return list of result dicts.
    """
    df = pd.merge(df_prop, df_emb, on="mof_uri", how="inner")
    log.info("  %s: merged → %d MOFs", name, len(df))
    if df.empty:
        return []

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    target_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in METADATA_COLS and c not in emb_cols
    ]
    log.info("    %d embedding dims, %d numeric targets", len(emb_cols), len(target_cols))

    regressors = make_regressors()
    results: List[dict] = []

    for target in target_cols:
        mask = df[target].notna()
        n = mask.sum()
        if n < MIN_SAMPLES:
            continue

        X = df.loc[mask, emb_cols].values.astype(np.float32)
        y = df.loc[mask, target].values.astype(np.float64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, random_state=SEED)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        for reg_name, reg_tmpl in regressors.items():
            reg = clone(reg_tmpl)
            try:
                reg.fit(X_tr_s, y_tr)
                preds = reg.predict(X_te_s)
                r2   = r2_score(y_te, preds)
                rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
            except Exception as e:
                log.warning("    %s / %s / %s failed: %s", name, target, reg_name, e)
                continue

            results.append({
                "Embedding": name,
                "Target":    target,
                "Model":     reg_name,
                "N_samples": int(n),
                "R2":        float(r2),
                "RMSE":      rmse,
            })

        log.info("    %-45s  n=%5d  best_R2=%.3f",
                 target[:45], n,
                 max(r["R2"] for r in results
                     if r["Embedding"] == name and r["Target"] == target))

    return results


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════

def make_per_method_heatmap(df_res: pd.DataFrame, method: str, out_dir: str):
    """Per-method R² heatmap (targets × models)."""
    sub = df_res[df_res["Embedding"] == method].copy()
    if sub.empty:
        return

    # Best model per target (keep for clean display)
    # Show MLP and RF as representative columns
    sub_best = sub.pivot_table(
        index="Target", columns="Model", values="R2", aggfunc="mean")

    # Clip to [-0.2, 1.0] for colour scale
    sub_best_clipped = sub_best.clip(lower=-0.2)

    fig_h = max(8, len(sub_best) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    sns.heatmap(
        sub_best_clipped, annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=-0.2, vmax=1.0,
        linewidths=0.4, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(f"{method} Embeddings: Chemical Property Prediction R²", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"heatmap_{method.lower()}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path)


def make_comparison_barplot(df_res: pd.DataFrame, out_dir: str, top_n: int = 20):
    """Cross-method comparison: top-N targets by best R² across all methods."""
    # Best R² per (Method, Target)
    best = (df_res
            .groupby(["Embedding", "Target"])["R2"]
            .max()
            .reset_index())

    # Rank targets by their best R² across ALL methods
    target_max = best.groupby("Target")["R2"].max().sort_values(ascending=False)
    top_targets = target_max.head(top_n).index.tolist()

    sub = best[best["Target"].isin(top_targets)]
    # Shorten long names
    sub = sub.copy()
    sub["TargetShort"] = sub["Target"].str[:35]

    fig, ax = plt.subplots(figsize=(16, 8))
    order = (sub.groupby("TargetShort")["R2"]
               .max()
               .sort_values(ascending=False)
               .index.tolist())
    sns.barplot(
        data=sub, x="TargetShort", y="R2", hue="Embedding",
        order=order, ax=ax, palette="Set2",
    )
    ax.set_title(f"Top-{top_n} Properties: Best R² per Embedding Method", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("R²")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(title="Embedding")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "comparison_top_properties.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path)


def make_model_comparison_heatmap(df_res: pd.DataFrame, out_dir: str):
    """Mean R² heatmap: embedding × model."""
    mean_r2 = (df_res
               .groupby(["Embedding", "Model"])["R2"]
               .mean()
               .reset_index())
    pivot = mean_r2.pivot(index="Model", columns="Embedding", values="R2")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Mean R² across all targets  (Embedding × Model)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "mean_r2_overview.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info("  Saved %s", out_path)


def make_scatter_plots(
    df_res: pd.DataFrame,
    emb_dfs: Dict[str, pd.DataFrame],
    df_prop: pd.DataFrame,
    out_dir: str,
    n_top: int = 12,
):
    """
    Actual-vs-predicted scatter plots for the top-n_top targets
    (by best R²), using the best model for each target×method combination.
    One panel per (method, target) arranged in a grid.
    """
    log.info("  Generating actual vs predicted scatter plots …")

    # Find top targets across all methods
    best_per_target = (df_res
                       .groupby("Target")["R2"]
                       .max()
                       .sort_values(ascending=False))
    top_targets = best_per_target.head(n_top).index.tolist()

    methods = list(emb_dfs.keys())
    ncols = len(methods)
    nrows = len(top_targets)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    regressors = make_regressors()

    for row_i, target in enumerate(top_targets):
        for col_j, method in enumerate(methods):
            ax = axes[row_i][col_j]

            df_merged = pd.merge(df_prop, emb_dfs[method], on="mof_uri", how="inner")
            mask = df_merged[target].notna()
            n = mask.sum()

            if n < MIN_SAMPLES:
                ax.set_visible(False)
                continue

            emb_cols = [c for c in df_merged.columns if c.startswith("emb_")]
            X = df_merged.loc[mask, emb_cols].values.astype(np.float32)
            y = df_merged.loc[mask, target].values.astype(np.float64)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_FRAC, random_state=SEED)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s  = scaler.transform(X_te)

            # Find the best model for this (method, target) from results
            sub = df_res[
                (df_res["Embedding"] == method) & (df_res["Target"] == target)]
            if sub.empty:
                ax.set_visible(False)
                continue
            best_model_name = sub.loc[sub["R2"].idxmax(), "Model"]
            r2_val          = sub["R2"].max()

            reg = clone(regressors[best_model_name])
            reg.fit(X_tr_s, y_tr)
            preds = reg.predict(X_te_s)

            ax.scatter(y_te, preds, alpha=0.3, s=6, edgecolors="none",
                       color=plt.cm.Set2(col_j))
            mn = min(y_te.min(), preds.min())
            mx = max(y_te.max(), preds.max())
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
            ax.set_xlabel("Actual", fontsize=8)
            ax.set_ylabel("Predicted", fontsize=8)
            title = f"{method}\n{target[:30]}\n(R²={r2_val:.3f}, n={n})"
            ax.set_title(title, fontsize=7)
            ax.tick_params(labelsize=7)

    plt.suptitle("Actual vs Predicted — Top Properties", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "scatter_top_properties.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path)


def make_binding_energy_detail(
    df_res: pd.DataFrame,
    emb_dfs: Dict[str, pd.DataFrame],
    df_prop: pd.DataFrame,
    out_dir: str,
):
    """Dedicated scatter plots for CO2 / H2O binding energy."""
    binding_targets = [
        t for t in df_prop.columns if "binding" in t.lower()
    ]
    if not binding_targets:
        return

    log.info("  Generating binding-energy detail plots …")
    methods = list(emb_dfs.keys())
    ncols = len(methods)
    nrows = len(binding_targets)
    if nrows == 0:
        return

    regressors = make_regressors()
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for row_i, target in enumerate(binding_targets):
        for col_j, method in enumerate(methods):
            ax = axes[row_i][col_j]
            df_merged = pd.merge(df_prop, emb_dfs[method], on="mof_uri", how="inner")
            mask = df_merged[target].notna()
            n = mask.sum()

            if n < MIN_SAMPLES:
                ax.text(0.5, 0.5, f"n={n} < {MIN_SAMPLES}",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{method}\n{target}", fontsize=8)
                continue

            emb_cols = [c for c in df_merged.columns if c.startswith("emb_")]
            X = df_merged.loc[mask, emb_cols].values.astype(np.float32)
            y = df_merged.loc[mask, target].values.astype(np.float64)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_FRAC, random_state=SEED)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s  = scaler.transform(X_te)

            # Try all models, pick best
            best_r2 = -999
            best_preds = None
            for reg_name, reg_tmpl in regressors.items():
                reg = clone(reg_tmpl)
                try:
                    reg.fit(X_tr_s, y_tr)
                    p = reg.predict(X_te_s)
                    r2_val = r2_score(y_te, p)
                    if r2_val > best_r2:
                        best_r2 = r2_val
                        best_preds = p
                        best_name = reg_name
                except Exception:
                    pass

            ax.scatter(y_te, best_preds, alpha=0.4, s=8,
                       color=plt.cm.Set2(col_j), edgecolors="none")
            mn = min(y_te.min(), best_preds.min())
            mx = max(y_te.max(), best_preds.max())
            ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
            ax.set_xlabel("Actual (eV)", fontsize=9)
            ax.set_ylabel("Predicted (eV)", fontsize=9)
            ax.set_title(
                f"{method}\n{target}\n(best: {best_name}, R²={best_r2:.3f}, n={n})",
                fontsize=8)
            ax.tick_params(labelsize=8)

    plt.suptitle("Binding Energy Prediction Detail", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "binding_energy_detail.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path)


def make_free_energy_detail(
    df_res: pd.DataFrame,
    emb_dfs: Dict[str, pd.DataFrame],
    df_prop: pd.DataFrame,
    out_dir: str,
):
    """Scatter plots for Free Energy (atom)."""
    target = "Free Energy (atom) [eV/atom]"
    if target not in df_prop.columns:
        log.warning("  Free Energy column not found — skipping detail plot")
        return

    log.info("  Generating free-energy detail plots …")
    methods = list(emb_dfs.keys())
    regressors = make_regressors()

    fig, axes = plt.subplots(1, len(methods),
                              figsize=(5 * len(methods), 5), squeeze=False)

    for col_j, method in enumerate(methods):
        ax = axes[0][col_j]
        df_merged = pd.merge(df_prop, emb_dfs[method], on="mof_uri", how="inner")
        mask = df_merged[target].notna()
        n = mask.sum()

        if n < MIN_SAMPLES:
            ax.text(0.5, 0.5, f"n={n} < {MIN_SAMPLES}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        emb_cols = [c for c in df_merged.columns if c.startswith("emb_")]
        X = df_merged.loc[mask, emb_cols].values.astype(np.float32)
        y = df_merged.loc[mask, target].values.astype(np.float64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, random_state=SEED)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        best_r2 = -999
        best_preds = None
        best_name = ""
        for reg_name, reg_tmpl in regressors.items():
            reg = clone(reg_tmpl)
            try:
                reg.fit(X_tr_s, y_tr)
                p = reg.predict(X_te_s)
                r2_val = r2_score(y_te, p)
                if r2_val > best_r2:
                    best_r2 = r2_val
                    best_preds = p
                    best_name = reg_name
            except Exception:
                pass

        ax.scatter(y_te, best_preds, alpha=0.3, s=6,
                   color=plt.cm.Set2(col_j), edgecolors="none")
        mn, mx = min(y_te.min(), best_preds.min()), max(y_te.max(), best_preds.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=0.8)
        ax.set_xlabel("Actual (eV/atom)", fontsize=10)
        ax.set_ylabel("Predicted (eV/atom)", fontsize=10)
        ax.set_title(
            f"{method}\nFree Energy (atom)\n(best: {best_name}, R²={best_r2:.3f}, n={n})",
            fontsize=9)

    plt.suptitle("Free Energy (atom) Prediction", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "free_energy_detail.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", out_path)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Chemical Property Evaluation  —  output: %s", OUT_DIR)
    log.info("=" * 70)

    # ── 1. Load chemical property table ──
    log.info("\n── Step 1: Load chemical properties ──")
    df_prop = pd.read_csv(CHEM_PATH)
    log.info("  Chem CSV: %d MOFs × %d columns", *df_prop.shape)

    # ── 2. Extract Free Energy from KG and merge ──
    log.info("\n── Step 2: Extract Free Energy (atom) from KG ──")
    df_fe = extract_free_energy_from_kg(KG_PATH)
    df_prop = pd.merge(df_prop, df_fe, on="mof_uri", how="left")
    fe_col = "Free Energy (atom) [eV/atom]"
    log.info("  Merged Free Energy: %d MOFs have value", df_prop[fe_col].notna().sum())

    # ── 3. Load and normalise embeddings ──
    log.info("\n── Step 3: Load and L2-normalise embeddings ──")
    emb_dfs: Dict[str, pd.DataFrame] = {
        "CompGCN":  load_compgcn(COMPGCN_CSV),
        "TransE":   load_transe(TRANSE_CSV),
        "Node2Vec": load_node2vec_256d(N2V_PT),
    }

    # ── 4. Run regression ──
    log.info("\n── Step 4: Regression evaluation ──")
    all_results: List[dict] = []
    for method, df_emb in emb_dfs.items():
        log.info("\n  [%s]", method)
        results = evaluate_embedding(method, df_emb, df_prop)
        all_results.extend(results)
        log.info("  → %d result rows", len(results))

    df_res = pd.DataFrame(all_results)
    csv_out = os.path.join(OUT_DIR, "chem_eval_results.csv")
    df_res.to_csv(csv_out, index=False)
    log.info("\n  Saved results: %s  (%d rows)", csv_out, len(df_res))

    # ── 5. Visualisations ──
    log.info("\n── Step 5: Visualisations ──")

    # 5a. Per-method heatmaps
    for method in emb_dfs:
        make_per_method_heatmap(df_res, method, OUT_DIR)

    # 5b. Cross-method comparison bar chart (top-20 targets)
    make_comparison_barplot(df_res, OUT_DIR, top_n=20)

    # 5c. Overview mean-R² heatmap (method × model)
    make_model_comparison_heatmap(df_res, OUT_DIR)

    # 5d. Actual-vs-predicted scatters for top-12 properties
    make_scatter_plots(df_res, emb_dfs, df_prop, OUT_DIR, n_top=12)

    # 5e. Binding energy detail
    make_binding_energy_detail(df_res, emb_dfs, df_prop, OUT_DIR)

    # 5f. Free Energy detail
    make_free_energy_detail(df_res, emb_dfs, df_prop, OUT_DIR)

    # ── 6. Summary table ──
    log.info("\n── Step 6: Summary table ──")
    summary = []
    for method in emb_dfs:
        sub = df_res[df_res["Embedding"] == method]
        if sub.empty:
            continue
        best_per_tgt = sub.groupby("Target")["R2"].max()
        summary.append({
            "Method":         method,
            "N_targets":      len(best_per_tgt),
            "Mean_best_R2":   round(best_per_tgt.mean(), 4),
            "Median_best_R2": round(best_per_tgt.median(), 4),
            "N_R2>0.3":       int((best_per_tgt > 0.3).sum()),
            "N_R2>0.5":       int((best_per_tgt > 0.5).sum()),
            "Best_target":    best_per_tgt.idxmax(),
            "Best_R2":        round(best_per_tgt.max(), 4),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "chem_eval_summary.csv"), index=False)
    log.info("\n%s", summary_df.to_string(index=False))

    elapsed = (time.time() - t0) / 60
    log.info("\n" + "=" * 70)
    log.info("COMPLETE  —  %.1f minutes elapsed  |  output: %s", elapsed, OUT_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
