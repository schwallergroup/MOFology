#!/usr/bin/env python3
"""
Chemical property prediction comparison:
  - ChemOnly
  - KGOnly (CompGCN / TransE / Node2Vec)
  - Hybrid (Chem + KG) for each method
"""

import argparse
import logging
import os
import re
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except Exception:  # pragma: no cover
    HAS_RDKIT = False

try:
    from pymatgen.core import Composition
    HAS_PYMATGEN = True
except Exception:  # pragma: no cover
    HAS_PYMATGEN = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="plasma")


def parse_args():
    parser = argparse.ArgumentParser(description="Chem property KG/Chem/Hybrid comparison")
    parser.add_argument("--chem_csv", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv"))
    parser.add_argument("--kg_path", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl"))
    parser.add_argument(
        "--compgcn_csv",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_compgcn_embeddings_256d_3layers.csv),
    )
    parser.add_argument(
        "--transe_csv",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings"/mof_transe_embeddings_256d.csv),
    )
    parser.add_argument(
        "--node2vec_pt",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_256d_p1.0_q1.0.pt),
    )
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/chem_combo_compare))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_samples", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_splits", type=int, default=2, help="Repeated random train/test splits for uncertainty estimates.")
    parser.add_argument("--viz_update_every", type=int, default=1, help="Refresh visualizations every N processed targets.")
    return parser.parse_args()


def _ensure_mof_uri_index(df: pd.DataFrame) -> pd.DataFrame:
    if "mof_uri" in df.columns:
        return df.set_index("mof_uri")
    if "uri" in df.columns:
        return df.set_index("uri")
    first = df.columns[0]
    if df[first].dtype == object:
        return df.set_index(first)
    raise ValueError("Could not identify mof_uri column")


def load_embedding_csv(path: str) -> pd.DataFrame:
    df = _ensure_mof_uri_index(pd.read_csv(path))
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        emb_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = df[emb_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    out.index.name = "mof_uri"
    return out


def load_node2vec_pt(path: str) -> pd.DataFrame:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    all_emb = saved["embeddings"]
    ent2id = saved["ent2id"]
    rows = []
    for uri, idx in ent2id.items():
        rows.append([uri] + all_emb[idx].tolist())
    cols = ["mof_uri"] + [f"emb_{i}" for i in range(all_emb.shape[1])]
    return pd.DataFrame(rows, columns=cols).set_index("mof_uri")


def _formula_features(formula: str, elements: List[str]) -> Dict[str, float]:
    if not isinstance(formula, str) or not formula:
        return {f"frac_{e}": 0.0 for e in elements}
    if HAS_PYMATGEN:
        try:
            comp = Composition(formula)
            total = float(sum(comp.values()))
            el_dict = comp.get_el_amt_dict()
            return {
                f"frac_{e}": float(el_dict.get(e, 0.0) / total) if total > 0 else 0.0
                for e in elements
            }
        except Exception:
            pass
    parsed: Dict[str, float] = {}
    total = 0.0
    for el, raw in re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula):
        v = float(raw) if raw else 1.0
        parsed[el] = parsed.get(el, 0.0) + v
        total += v
    if total <= 0:
        return {f"frac_{e}": 0.0 for e in elements}
    return {f"frac_{e}": float(parsed.get(e, 0.0) / total) for e in elements}


def _smiles_fp(smiles_field: str, n_bits: int = 256) -> np.ndarray:
    if not HAS_RDKIT or not isinstance(smiles_field, str) or not smiles_field.strip():
        return np.zeros(n_bits, dtype=np.float32)
    fps = []
    for smi in [x.strip() for x in smiles_field.split(";") if x.strip()]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        fps.append(np.asarray(fp, dtype=np.float32))
    if not fps:
        return np.zeros(n_bits, dtype=np.float32)
    return np.mean(np.stack(fps), axis=0).astype(np.float32)


def build_chem_feature_df(df_prop: pd.DataFrame) -> pd.DataFrame:
    df = df_prop.drop_duplicates(subset=["mof_uri"]).copy()
    numeric_cols = [c for c in [
        "Number of atoms",
        "Unit cell volume",
        "Space group number",
        "Density",
        "Largest cavity diameter",
        "Pore limiting diameter",
        "Band gap (PBE)",
    ] if c in df.columns]
    df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=df.index)

    cat_cols = [c for c in ["crystal_system", "space_group", "topology", "metal_cluster_elements"] if c in df.columns]
    # Limit one-hot encoding to top-20 categories per column to prevent feature explosion
    if cat_cols:
        df_cat_parts = []
        for col in cat_cols:
            top_vals = df[col].value_counts().head(20).index.tolist()
            temp = df[col].fillna("UNK").apply(lambda x: x if x in top_vals else "OTHER")
            df_cat_parts.append(pd.get_dummies(temp, prefix=f"cat_{col}", dtype=np.float32))
        df_cat = pd.concat(df_cat_parts, axis=1)
    else:
        df_cat = pd.DataFrame(index=df.index)

    elements = ["C", "H", "N", "O", "S", "Zn", "Cu", "Fe", "Al", "Zr", "Co", "Ni"]
    if "chemical_formula" in df.columns:
        df_formula = pd.DataFrame(df["chemical_formula"].fillna("").apply(lambda x: _formula_features(x, elements)).tolist())
    else:
        df_formula = pd.DataFrame([{f"frac_{e}": 0.0 for e in elements}] * len(df))

    if "linker_smiles" in df.columns:
        fp = np.vstack(df["linker_smiles"].fillna("").map(_smiles_fp).values)
    else:
        fp = np.zeros((len(df), 256), dtype=np.float32)
    df_fp = pd.DataFrame(fp, columns=[f"fp_linker_{i}" for i in range(fp.shape[1])])

    out = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True), df_formula, df_fp], axis=1)
    out.insert(0, "mof_uri", df["mof_uri"].values)
    return out.set_index("mof_uri")


def preprocess(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    return scaler.fit_transform(X_train_i), scaler.transform(X_test_i)


def prepare_xy(
    target: str,
    family: str,
    method: str,
    df_prop: pd.DataFrame,
    chem_features: pd.DataFrame,
    embeddings: Dict[str, pd.DataFrame],
) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    """
    Build (idx, X, y) for a target/family/method.
    Applies anti-leakage rule by dropping target from chem feature table.
    """
    y = pd.to_numeric(df_prop[target], errors="coerce")
    keep = df_prop.loc[y.notna(), "mof_uri"].tolist()
    chem_no_leak = chem_features.drop(columns=[target], errors="ignore")

    if family == "ChemOnly":
        idx = chem_no_leak.index.intersection(keep)
        X = chem_no_leak.loc[idx].values.astype(np.float32)
    elif family == "KGOnly":
        emb_df = embeddings[method]
        idx = emb_df.index.intersection(keep)
        X = emb_df.loc[idx].values.astype(np.float32)
    else:
        emb_df = embeddings[method]
        idx = emb_df.index.intersection(chem_no_leak.index).intersection(keep)
        X = np.hstack(
            [emb_df.loc[idx].values.astype(np.float32), chem_no_leak.loc[idx].values.astype(np.float32)]
        )

    y_vals = df_prop.set_index("mof_uri").loc[idx, target].values.astype(np.float64)
    return idx, X, y_vals


def regressors(seed: int):
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=seed
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
    return models


def evaluate_one(
    target: str,
    family: str,
    method: str,
    df_prop: pd.DataFrame,
    chem_features: pd.DataFrame,
    embeddings: Dict[str, pd.DataFrame],
    seed: int,
    test_size: float,
    n_splits: int,
) -> List[dict]:
    idx, X, y_vals = prepare_xy(target, family, method, df_prop, chem_features, embeddings)
    if len(idx) < 5:
        return []

    rows = []
    for split_id in range(n_splits):
        split_seed = seed + split_id
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_vals, test_size=test_size, random_state=split_seed
        )
        X_train, X_test = preprocess(X_train, X_test)

        for model_name, model_template in regressors(seed).items():
            model = clone(model_template)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rows.append(
                {
                    "Target": target,
                    "Embedding": method,
                    "FeatureFamily": family,
                    "Model": model_name,
                    "Split": split_id,
                    "Samples": int(len(idx)),
                    "R2": float(r2_score(y_test, preds)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
                }
            )
    return rows


def save_main_outputs(df_res: pd.DataFrame, out_dir: str) -> None:
    """Write metrics tables + heatmap + top-target barplot with CIs."""
    if df_res.empty:
        return

    out_csv = os.path.join(out_dir, "chem_combo_results.csv")
    df_res.to_csv(out_csv, index=False)

    best = (
        df_res.groupby(["Target", "Embedding", "FeatureFamily"], as_index=False)["R2"]
        .mean()
        .sort_values(["Target", "R2"], ascending=[True, False])
    )
    best.to_csv(os.path.join(out_dir, "chem_combo_best_r2.csv"), index=False)

    mean_r2 = (
        df_res.groupby(["Embedding", "FeatureFamily", "Model"], as_index=False)["R2"]
        .mean()
    )
    mean_r2["Method"] = mean_r2["Embedding"] + "|" + mean_r2["FeatureFamily"]
    pivot = mean_r2.pivot(index="Model", columns="Method", values="R2")
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="plasma", ax=ax)
    ax.set_title("Chemical Prediction Mean R2 (KG/Chem/Hybrid)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chem_combo_mean_r2_heatmap.png"), dpi=260)
    plt.close(fig)

    top_targets = (
        df_res.groupby("Target")["R2"].max().sort_values(ascending=False).head(12).index.tolist()
    )
    sub = df_res[df_res["Target"].isin(top_targets)].copy()
    sub["Method"] = sub["Embedding"] + "|" + sub["FeatureFamily"]
    sub_plot = (
        sub.groupby(["Target", "Method", "Split"], as_index=False)["R2"]
        .max()
    )
    fig, ax = plt.subplots(figsize=(15, 7))
    pal = sns.color_palette("plasma", n_colors=sub_plot["Method"].nunique())
    sns.barplot(
        data=sub_plot,
        x="Target",
        y="R2",
        hue="Method",
        ax=ax,
        palette=pal,
        errorbar=("ci", 95),
    )
    ax.set_title("Top Targets: R2 by Method/Feature Family")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "chem_combo_top_targets_barplot.png"), dpi=260)
    plt.close(fig)

    summary = (
        df_res.groupby(["Embedding", "FeatureFamily"], as_index=False)
        .agg(
            MeanBestR2=("R2", "mean"),
            StdBestR2=("R2", "std"),
            BestR2=("R2", "max"),
            MeanRMSE=("RMSE", "mean"),
            NumRows=("R2", "size"),
        )
        .sort_values("MeanBestR2", ascending=False)
    )
    summary.to_csv(os.path.join(out_dir, "chem_combo_summary.csv"), index=False)


def save_best_pred_vs_actual_plots(
    df_res: pd.DataFrame,
    df_prop: pd.DataFrame,
    chem_features: pd.DataFrame,
    embeddings: Dict[str, pd.DataFrame],
    seed: int,
    test_size: float,
    out_dir: str,
) -> None:
    """
    For each property, train best-performing (Embedding, FeatureFamily, Model)
    and save one predicted-vs-actual scatter plot.
    """
    if df_res.empty:
        return

    plot_dir = os.path.join(out_dir, "best_pred_vs_actual")
    os.makedirs(plot_dir, exist_ok=True)

    # Pick best config by mean R2 over splits for each target.
    best_cfg = (
        df_res.groupby(["Target", "Embedding", "FeatureFamily", "Model"], as_index=False)["R2"]
        .mean()
        .sort_values(["Target", "R2"], ascending=[True, False])
        .groupby("Target", as_index=False)
        .first()
    )
    best_cfg.to_csv(os.path.join(out_dir, "best_model_per_target.csv"), index=False)

    model_templates = regressors(seed)
    plasma = plt.get_cmap("plasma")

    for _, row in best_cfg.iterrows():
        target = str(row["Target"])
        method = str(row["Embedding"])
        family = str(row["FeatureFamily"])
        model_name = str(row["Model"])
        mean_r2 = float(row["R2"])

        idx, X, y_vals = prepare_xy(target, family, method, df_prop, chem_features, embeddings)
        if len(idx) < 5:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_vals, test_size=test_size, random_state=seed
        )
        X_train, X_test = preprocess(X_train, X_test)

        model_template = model_templates.get(model_name)
        if model_template is None:
            continue
        model = clone(model_template)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        this_r2 = float(r2_score(y_test, preds))

        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        ax.scatter(y_test, preds, s=12, alpha=0.35, color=plasma(0.72), edgecolors="none")
        lo = min(y_test.min(), preds.min())
        hi = max(y_test.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color=plasma(0.15))
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(
            f"{target}\nBest={method}|{family}|{model_name} "
            f"(mean split R2={mean_r2:.3f}, this split R2={this_r2:.3f})"
        )
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"pred_vs_actual_{_safe(target)}.png"), dpi=240)
        plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    df_prop = pd.read_csv(args.chem_csv)
    if "mof_uri" not in df_prop.columns:
        raise RuntimeError("chem_csv missing required column 'mof_uri'")
    log.info("Loaded properties: %d rows", len(df_prop))

    chem_features = build_chem_feature_df(df_prop)
    log.info("Chem feature matrix: %d MOFs x %d dims", chem_features.shape[0], chem_features.shape[1])

    embeddings = {
        "CompGCN": load_embedding_csv(args.compgcn_csv),
        "TransE": load_embedding_csv(args.transe_csv),
        "Node2Vec": load_node2vec_pt(args.node2vec_pt),
    }
    log.info(
        "Run settings: n_splits=%d, min_samples=%d, test_size=%.2f, viz_update_every=%d",
        args.n_splits, args.min_samples, args.test_size, args.viz_update_every
    )
    for name, df_emb in embeddings.items():
        log.info("%s embeddings: %d MOFs x %d dims", name, df_emb.shape[0], df_emb.shape[1])

    metadata_cols = {
        "mof_uri", "csd_code", "chemical_formula", "mofid",
        "topology", "metal_cluster_elements", "linker_smiles",
        "space_group", "crystal_system",
    }
    target_cols = [
        c for c in df_prop.select_dtypes(include=[np.number]).columns
        if c not in metadata_cols
    ]
    log.info("Candidate targets: %d", len(target_cols))

    all_rows: List[dict] = []
    processed_targets = 0
    for target in target_cols:
        n = pd.to_numeric(df_prop[target], errors="coerce").notna().sum()
        if n < args.min_samples:
            continue
        chem_only_rows = evaluate_one(
            target, "ChemOnly", "ChemOnly", df_prop, chem_features, embeddings, args.seed, args.test_size, args.n_splits
        )
        all_rows.extend(chem_only_rows)
        for method in embeddings:
            all_rows.extend(evaluate_one(target, "KGOnly", method, df_prop, chem_features, embeddings, args.seed, args.test_size, args.n_splits))
            all_rows.extend(evaluate_one(target, "Hybrid", method, df_prop, chem_features, embeddings, args.seed, args.test_size, args.n_splits))
        processed_targets += 1
        log.info("Processed target: %s (n=%d)", target, n)
        if processed_targets % max(args.viz_update_every, 1) == 0:
            save_main_outputs(pd.DataFrame(all_rows), args.out_dir)
            log.info("Updated visualizations/results after %d targets.", processed_targets)

    if not all_rows:
        raise RuntimeError("No results produced; check overlaps and min_samples.")

    df_res = pd.DataFrame(all_rows)
    save_main_outputs(df_res, args.out_dir)
    save_best_pred_vs_actual_plots(
        df_res=df_res,
        df_prop=df_prop,
        chem_features=chem_features,
        embeddings=embeddings,
        seed=args.seed,
        test_size=args.test_size,
        out_dir=args.out_dir,
    )
    log.info("Saved outputs to %s", args.out_dir)


if __name__ == "__main__":
    main()
