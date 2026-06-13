#!/usr/bin/env python3
"""
Compare structural-label imputation using:
  - KG-only features (CompGCN / TransE / Node2Vec)
  - Chem-only features
  - Hybrid features (KG || Chem)
"""

import argparse
import logging
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdflib import Graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

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


def parse_args():
    parser = argparse.ArgumentParser(description="MOF imputation comparison study")
    parser.add_argument("--kg_path", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl"))
    parser.add_argument("--chem_csv", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv"))
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
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/imputation_compare))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_class_samples", type=int, default=10)
    parser.add_argument("--min_total_samples", type=int, default=50)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv_folds", type=int, default=1,
                        help="If > 1, use stratified K-fold CV; emit one metrics row per fold.")
    return parser.parse_args()


def get_structural_labels_from_kg(kg_path: str) -> pd.DataFrame:
    g = Graph()
    g.parse(kg_path, format="turtle")
    query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?topology ?metal WHERE {
        ?mof a mof:MOF .
        OPTIONAL {
            ?mof mof:hasTopology ?topoNode .
            ?topoNode mof:topologyCode ?topology .
        }
        OPTIONAL {
            ?mof mof:hasMetalNode ?node .
            ?node mof:hasMetalElement ?metal .
        }
    }
    """
    rows = []
    for row in g.query(query):
        rows.append(
            {
                "mof_uri": str(row.mof),
                "topology": str(row.topology) if row.topology else None,
                "metal_element": str(row.metal) if row.metal else None,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["topology", "metal_element"], how="all")
    df = (
        df.groupby("mof_uri")
        .agg(
            {
                "topology": "first",
                "metal_element": lambda x: sorted(list(set(x.dropna())))[0] if len(x.dropna()) > 0 else None,
            }
        )
        .reset_index()
    )
    return df


def _ensure_mof_uri_index(df: pd.DataFrame) -> pd.DataFrame:
    if "mof_uri" in df.columns:
        return df.set_index("mof_uri")
    if "uri" in df.columns:
        return df.set_index("uri")
    first = df.columns[0]
    if df[first].dtype == object:
        return df.set_index(first)
    raise ValueError("Could not identify MOF URI index column.")


def load_embedding_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_mof_uri_index(df)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        emb_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = df[emb_cols].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(axis=1, how="all")
    out.index.name = "mof_uri"
    return out


def load_node2vec_pt(path: str) -> pd.DataFrame:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    emb = saved["embeddings"]
    ent2id = saved["ent2id"]
    rows = []
    for uri, idx in ent2id.items():
        rows.append([uri] + emb[idx].tolist())
    cols = ["mof_uri"] + [f"emb_{i}" for i in range(emb.shape[1])]
    df = pd.DataFrame(rows, columns=cols).set_index("mof_uri")
    return df


def _formula_fractions(formula: str, elements: List[str]) -> Dict[str, float]:
    if not isinstance(formula, str) or not formula:
        return {f"frac_{e}": 0.0 for e in elements}
    if HAS_PYMATGEN:
        try:
            comp = Composition(formula)
            total = float(sum(comp.values()))
            el_dict = comp.get_el_amt_dict()
            return {f"frac_{e}": float(el_dict.get(e, 0.0) / total) if total > 0 else 0.0 for e in elements}
        except Exception:
            pass
    counts = dict(re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula))
    parsed = {}
    total = 0.0
    for e, raw_v in counts.items():
        v = float(raw_v) if raw_v else 1.0
        parsed[e] = parsed.get(e, 0.0) + v
        total += v
    if total <= 0:
        return {f"frac_{e}": 0.0 for e in elements}
    return {f"frac_{e}": float(parsed.get(e, 0.0) / total) for e in elements}


def _smiles_fingerprint(smiles_field: str, n_bits: int = 256) -> np.ndarray:
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


def build_chem_features(df_prop: pd.DataFrame, mof_uris: List[str]) -> pd.DataFrame:
    df = df_prop[df_prop["mof_uri"].isin(set(mof_uris))].copy()
    df = df.drop_duplicates(subset=["mof_uri"])
    if df.empty:
        return pd.DataFrame()

    numeric_candidates = [
        "Number of atoms",
        "Unit cell volume",
        "Space group number",
        "Density",
        "Largest cavity diameter",
        "Pore limiting diameter",
        "Band gap (PBE)",
    ]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=df.index)

    cat_candidates = ["crystal_system", "space_group", "topology", "metal_cluster_elements"]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    if cat_cols:
        df_cat = pd.get_dummies(df[cat_cols].fillna("UNK"), prefix=[f"cat_{c}" for c in cat_cols], dtype=np.float32)
    else:
        df_cat = pd.DataFrame(index=df.index)

    elements = ["C", "H", "N", "O", "S", "Zn", "Cu", "Fe", "Al", "Zr", "Co", "Ni"]
    formula_col = "chemical_formula" if "chemical_formula" in df.columns else None
    if formula_col:
        formula_feats = pd.DataFrame(df[formula_col].fillna("").apply(lambda x: _formula_fractions(x, elements)).tolist())
    else:
        formula_feats = pd.DataFrame([{f"frac_{e}": 0.0 for e in elements}] * len(df))

    if "linker_smiles" in df.columns:
        linker_fp = np.vstack(df["linker_smiles"].fillna("").map(_smiles_fingerprint).values)
    else:
        linker_fp = np.zeros((len(df), 256), dtype=np.float32)
    fp_cols = [f"fp_linker_{i}" for i in range(linker_fp.shape[1])]
    df_fp = pd.DataFrame(linker_fp, columns=fp_cols)

    features = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True), formula_feats, df_fp], axis=1)
    features.insert(0, "mof_uri", df["mof_uri"].values)
    return features.set_index("mof_uri")


def preprocess_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)
    return scaler.fit_transform(X_train_i), scaler.transform(X_test_i)


def evaluate_task(
    method: str,
    family: str,
    target: str,
    y_df: pd.DataFrame,
    emb_df: Optional[pd.DataFrame],
    chem_df: pd.DataFrame,
    min_class_samples: int,
    min_total_samples: int,
    test_size: float,
    seed: int,
    cv_folds: int = 1,
):
    y = y_df[["mof_uri", target]].dropna().set_index("mof_uri")[target]
    counts = y.value_counts()
    valid_classes = counts[counts >= min_class_samples].index
    y = y[y.isin(valid_classes)]
    if len(y) < min_total_samples:
        return None

    if family == "kg":
        if emb_df is None or emb_df.empty:
            return None
        common = y.index.intersection(emb_df.index)
        X = emb_df.loc[common].values.astype(np.float32)
        y_use = y.loc[common]
    elif family == "chem":
        common = y.index.intersection(chem_df.index)
        X = chem_df.loc[common].values.astype(np.float32)
        y_use = y.loc[common]
    else:
        if emb_df is None or emb_df.empty:
            return None
        common = y.index.intersection(emb_df.index).intersection(chem_df.index)
        X = np.hstack(
            [
                emb_df.loc[common].values.astype(np.float32),
                chem_df.loc[common].values.astype(np.float32),
            ]
        )
        y_use = y.loc[common]

    if len(y_use) < min_total_samples:
        return None

    # Re-filter after feature intersection because rare classes can reappear
    # when a subset of MOFs is available for a given feature family.
    post_counts = y_use.value_counts()
    keep_classes = post_counts[post_counts >= 2].index
    y_use = y_use[y_use.isin(keep_classes)]
    if len(y_use) < min_total_samples or y_use.nunique() < 2:
        return None

    # Keep X aligned with y_use index order after post-filtering.
    if family == "kg":
        X = emb_df.loc[y_use.index].values.astype(np.float32)
    elif family == "chem":
        X = chem_df.loc[y_use.index].values.astype(np.float32)
    else:
        X = np.hstack(
            [
                emb_df.loc[y_use.index].values.astype(np.float32),
                chem_df.loc[y_use.index].values.astype(np.float32),
            ]
        )

    le = LabelEncoder()
    y_enc = le.fit_transform(y_use.values)
    n_classes = len(np.unique(y_enc))

    if cv_folds and cv_folds > 1:
        min_per_class = int(pd.Series(y_enc).value_counts().min())
        n_splits = min(cv_folds, max(2, min_per_class))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y_enc))
    else:
        n_test = int(np.ceil(len(y_enc) * test_size))
        use_stratify = y_enc if n_test >= n_classes else None
        tr, te = train_test_split(
            np.arange(len(y_enc)), test_size=test_size, random_state=seed, stratify=use_stratify
        )
        splits = [(tr, te)]

    fold_metrics: List[dict] = []
    for fold_idx, (idx_train, idx_test) in enumerate(splits):
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y_enc[idx_train], y_enc[idx_test]
        X_train, X_test = preprocess_train_test(X_train, X_test)
        clf = RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=seed + fold_idx,
            class_weight="balanced_subsample",
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        base = Counter(y_train).most_common(1)[0][0]
        dummy = np.full_like(y_test, base)
        fold_metrics.append({
            "Embedding": method,
            "FeatureFamily": family,
            "Target": target,
            "Fold": int(fold_idx),
            "Samples": int(len(y_use)),
            "NumClasses": int(len(le.classes_)),
            "Accuracy": float(accuracy_score(y_test, preds)),
            "WeightedF1": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
            "MacroF1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "BaselineAccuracy": float(accuracy_score(y_test, dummy)),
            "_y_true": le.inverse_transform(y_test),
            "_y_pred": le.inverse_transform(preds),
        })
    return fold_metrics


def plot_confusion_matrices(results: List[dict], out_dir: str):
    cm_dir = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    for res in results:
        y_true = np.array(res["_y_true"])
        y_pred = np.array(res["_y_pred"])
        top_classes = [k for k, _ in Counter(y_true).most_common(10)]
        mask = np.isin(y_true, top_classes)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        if len(y_t) == 0:
            continue
        cm = confusion_matrix(y_t, y_p, labels=top_classes)
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=top_classes, yticklabels=top_classes, ax=ax)
        ax.set_title(f"{res['Target']} | {res['Embedding']} | {res['FeatureFamily']}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.tight_layout()
        fname = f"cm_{res['Target']}_{res['Embedding']}_{res['FeatureFamily']}.png".replace(" ", "_")
        fig.savefig(os.path.join(cm_dir, fname), dpi=250)
        plt.close(fig)


def plot_summary(df: pd.DataFrame, out_dir: str):
    for target in sorted(df["Target"].unique()):
        sub = df[df["Target"] == target].copy()
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=sub, x="Embedding", y="WeightedF1", hue="FeatureFamily", ax=ax)
        ax.set_title(f"{target}: Weighted F1 by Embedding and Feature Family")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bar_{target}_weighted_f1.png"), dpi=250)
        plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    log.info("Loading KG-derived structural labels...")
    df_labels = get_structural_labels_from_kg(args.kg_path)
    if df_labels.empty:
        raise RuntimeError("No structural labels extracted from KG.")
    log.info("Label table: %d MOFs", len(df_labels))

    log.info("Loading chemical properties and building chem features...")
    df_prop = pd.read_csv(args.chem_csv)
    chem_df = build_chem_features(df_prop, df_labels["mof_uri"].tolist())
    log.info("Chem feature table: %d MOFs x %d feats", chem_df.shape[0], chem_df.shape[1])

    emb_map: Dict[str, pd.DataFrame] = {
        "CompGCN": load_embedding_csv(args.compgcn_csv),
        "TransE": load_embedding_csv(args.transe_csv),
        "Node2Vec": load_node2vec_pt(args.node2vec_pt),
    }
    for k, v in emb_map.items():
        log.info("%s embeddings: %d MOFs x %d dims", k, v.shape[0], v.shape[1])

    results = []
    for target in ["topology", "metal_element"]:
        for method, emb_df in emb_map.items():
            for family in ["kg", "chem", "hybrid"]:
                res_list = evaluate_task(
                    method=method,
                    family=family,
                    target=target,
                    y_df=df_labels,
                    emb_df=emb_df,
                    chem_df=chem_df,
                    min_class_samples=args.min_class_samples,
                    min_total_samples=args.min_total_samples,
                    test_size=args.test_size,
                    seed=args.seed,
                    cv_folds=args.cv_folds,
                )
                if res_list:
                    accs = [r["Accuracy"] for r in res_list]
                    wf1s = [r["WeightedF1"] for r in res_list]
                    log.info(
                        "%s | %s | %s -> Acc %.4f +/- %.4f | wF1 %.4f +/- %.4f (n_folds=%d)",
                        method, family, target,
                        float(np.mean(accs)), float(np.std(accs)),
                        float(np.mean(wf1s)), float(np.std(wf1s)),
                        len(res_list),
                    )
                    results.extend(res_list)

    if not results:
        raise RuntimeError("No imputation results were generated.")

    plot_confusion_matrices(results, args.out_dir)
    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    metrics_df.to_csv(os.path.join(args.out_dir, "imputation_metrics_comparison.csv"), index=False)
    plot_summary(metrics_df, args.out_dir)

    agg_mean = metrics_df.groupby(
        ["Target", "Embedding", "FeatureFamily"], as_index=False
    )[["Accuracy", "WeightedF1", "MacroF1", "BaselineAccuracy"]].mean()
    agg_std = metrics_df.groupby(
        ["Target", "Embedding", "FeatureFamily"], as_index=False
    )[["Accuracy", "WeightedF1", "MacroF1"]].std().rename(
        columns={"Accuracy": "Accuracy_std", "WeightedF1": "WeightedF1_std", "MacroF1": "MacroF1_std"}
    )
    summary_df = agg_mean.merge(
        agg_std, on=["Target", "Embedding", "FeatureFamily"], how="left"
    ).sort_values(["Target", "WeightedF1"], ascending=[True, False])
    summary_df.to_csv(os.path.join(args.out_dir, "imputation_summary.csv"), index=False)
    log.info("Saved outputs to %s", args.out_dir)


if __name__ == "__main__":
    main()
