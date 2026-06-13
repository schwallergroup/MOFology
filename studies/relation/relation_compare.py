#!/usr/bin/env python3
"""
Functionalization delta prediction comparison:
  - KG-only features (CompGCN / TransE / Node2Vec)
  - Chem-only features
  - Hybrid features (KG || Chem)
Targets:
  - Delta_CO2 = child_CO2 - parent_CO2
  - Delta_H2O = child_H2O - parent_H2O
"""

import argparse
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdflib import Graph
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
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


def parse_args():
    parser = argparse.ArgumentParser(description="MOF relation comparison study")
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
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/relation_compare))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cv_folds", type=int, default=1,
                        help="If > 1, use K-fold CV; emit one metrics row per fold")
    parser.add_argument("--min_samples", type=int, default=80)
    parser.add_argument("--min_samples_per_amine", type=int, default=25)
    parser.add_argument(
        "--amine_types",
        default="",
        help="Optional comma-separated amine codes to evaluate (e.g., een,dmen,nmen). Empty means evaluate all with enough samples.",
    )
    return parser.parse_args()


def get_parent_child_pairs(kg_path: str) -> pd.DataFrame:
    g = Graph()
    g.parse(kg_path, format="turtle")
    query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    PREFIX syn: <http://emmo.info/domain-mof/synthesis#>
    SELECT DISTINCT ?parentMof ?funcMof ?propMof ?amineCode ?amineName ?parentCO2BE ?funcCO2BE ?parentH2OBE ?funcH2OBE
    WHERE {
      ?funcMof syn:hasFunctionalization ?func .
      ?func syn:hasFunctionalizationType syn:AmineFunctionalization .
      OPTIONAL { ?func syn:usesFunctionalGroup ?amineCode . }
      OPTIONAL { ?func syn:functionalGroupName ?amineName . }
      ?funcMof syn:derivedFrom ?parentMof .
      BIND(IRI(REPLACE(STR(?funcMof), "FuncMOF_", "MOF_")) AS ?propMof)
      OPTIONAL {
        {
          ?p1 mof:hasComputationalPropertyOwner ?propMof .
        } UNION {
          ?propMof mof:hasComputationalProperty ?p1 .
        }
        ?p1 mof:propertyName ?n1 ;
            mof:propertyValue ?funcCO2BE .
        FILTER(
          CONTAINS(LCASE(STR(?n1)), "co2") &&
          (CONTAINS(LCASE(STR(?n1)), "binding energy") || CONTAINS(LCASE(STR(?n1)), "binding"))
        )
      }
      OPTIONAL {
        {
          ?p2 mof:hasComputationalPropertyOwner ?parentMof .
        } UNION {
          ?parentMof mof:hasComputationalProperty ?p2 .
        }
        ?p2 mof:propertyName ?n2 ;
            mof:propertyValue ?parentCO2BE .
        FILTER(
          CONTAINS(LCASE(STR(?n2)), "co2") &&
          (CONTAINS(LCASE(STR(?n2)), "binding energy") || CONTAINS(LCASE(STR(?n2)), "binding"))
        )
      }
      OPTIONAL {
        {
          ?p3 mof:hasComputationalPropertyOwner ?propMof .
        } UNION {
          ?propMof mof:hasComputationalProperty ?p3 .
        }
        ?p3 mof:propertyName ?n3 ;
            mof:propertyValue ?funcH2OBE .
        FILTER(
          CONTAINS(LCASE(STR(?n3)), "h2o") &&
          (CONTAINS(LCASE(STR(?n3)), "binding energy") || CONTAINS(LCASE(STR(?n3)), "binding"))
        )
      }
      OPTIONAL {
        {
          ?p4 mof:hasComputationalPropertyOwner ?parentMof .
        } UNION {
          ?parentMof mof:hasComputationalProperty ?p4 .
        }
        ?p4 mof:propertyName ?n4 ;
            mof:propertyValue ?parentH2OBE .
        FILTER(
          CONTAINS(LCASE(STR(?n4)), "h2o") &&
          (CONTAINS(LCASE(STR(?n4)), "binding energy") || CONTAINS(LCASE(STR(?n4)), "binding"))
        )
      }
    }
    """
    rows = []
    for row in g.query(query):
        amine_code = str(row.amineCode) if row.amineCode else ""
        amine_code = amine_code.split("#")[-1] if amine_code else ""
        if amine_code.startswith("CHEM_"):
            amine_code = amine_code.replace("CHEM_", "", 1)
        amine_name = str(row.amineName).strip() if row.amineName else ""
        rows.append(
            {
                "parent_uri": str(row.parentMof),
                "child_uri": str(row.funcMof),
                "child_prop_uri": str(row.propMof),
                "amine_code": amine_code.lower() if amine_code else "unknown",
                "amine_name": amine_name if amine_name else "Unknown",
                "p_co2": float(row.parentCO2BE) if row.parentCO2BE else np.nan,
                "c_co2": float(row.funcCO2BE) if row.funcCO2BE else np.nan,
                "p_h2o": float(row.parentH2OBE) if row.parentH2OBE else np.nan,
                "c_h2o": float(row.funcH2OBE) if row.funcH2OBE else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Collapse duplicate rows; keep all amine-functionalized MOFs, including CHEM_None.
    keys = ["parent_uri", "child_uri", "child_prop_uri", "amine_code", "amine_name"]
    df = (
        df.groupby(keys, as_index=False)[["p_co2", "c_co2", "p_h2o", "c_h2o"]]
        .mean()
    )
    if df.empty:
        return df
    df["Delta_CO2"] = df["c_co2"] - df["p_co2"]
    df["Delta_H2O"] = df["c_h2o"] - df["p_h2o"]
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
    return pd.DataFrame(rows, columns=cols).set_index("mof_uri")


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
    parts = re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula)
    parsed: Dict[str, float] = {}
    total = 0.0
    for el, raw_n in parts:
        n = float(raw_n) if raw_n else 1.0
        parsed[el] = parsed.get(el, 0.0) + n
        total += n
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
    if "chemical_formula" in df.columns:
        formula_feats = pd.DataFrame(df["chemical_formula"].fillna("").apply(lambda x: _formula_fractions(x, elements)).tolist())
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


def make_regressors(seed: int):
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=seed
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
    return models


def evaluate_family_method(
    pairs_df: pd.DataFrame,
    method: str,
    family: str,
    emb_df: Optional[pd.DataFrame],
    chem_df: pd.DataFrame,
    test_size: float,
    seed: int,
    out_dir: str,
    amine_label: str = "ALL",
    cv_folds: int = 1,
) -> List[dict]:
    df = pairs_df.copy()
    if family == "kg":
        if emb_df is None or emb_df.empty:
            return []
        df = df[df["parent_uri"].isin(emb_df.index) & df["child_uri"].isin(emb_df.index)].copy()
        if df.empty:
            return []
        p_emb = emb_df.loc[df["parent_uri"]].values.astype(np.float32)
        c_emb = emb_df.loc[df["child_uri"]].values.astype(np.float32)
        amine_ohe = pd.get_dummies(df["amine_code"], prefix="amine", dtype=np.float32).values.astype(np.float32)
        X = np.hstack([p_emb, c_emb, c_emb - p_emb, amine_ohe])
    elif family == "chem":
        df = df[df["parent_uri"].isin(chem_df.index)].copy()
        if df.empty:
            return []
        amine_ohe = pd.get_dummies(df["amine_code"], prefix="amine", dtype=np.float32).values.astype(np.float32)
        X = np.hstack([chem_df.loc[df["parent_uri"]].values.astype(np.float32), amine_ohe])
    else:
        if emb_df is None or emb_df.empty:
            return []
        common = set(emb_df.index).intersection(set(chem_df.index))
        df = df[df["parent_uri"].isin(common) & df["child_uri"].isin(emb_df.index)].copy()
        if df.empty:
            return []
        p_emb = emb_df.loc[df["parent_uri"]].values.astype(np.float32)
        c_emb = emb_df.loc[df["child_uri"]].values.astype(np.float32)
        amine_ohe = pd.get_dummies(df["amine_code"], prefix="amine", dtype=np.float32).values.astype(np.float32)
        X = np.hstack(
            [
                p_emb,
                c_emb,
                c_emb - p_emb,
                chem_df.loc[df["parent_uri"]].values.astype(np.float32),
                amine_ohe,
            ]
        )
    if len(df) == 0:
        return []

    results = []
    regressors = make_regressors(seed)
    target_specs = [("Delta_CO2", "p_co2", "c_co2"), ("Delta_H2O", "p_h2o", "c_h2o")]
    for target_name, p_col, c_col in target_specs:
        valid = np.isfinite(df[p_col].values) & np.isfinite(df[c_col].values) & np.isfinite(df[target_name].values)
        if int(valid.sum()) < 20:
            continue
        X_t = X[valid]
        y_t = df.loc[valid, target_name].values.astype(np.float32)

        if cv_folds and cv_folds > 1:
            splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            splits = [(tr, te) for tr, te in splitter.split(np.arange(len(y_t)))]
        else:
            tr, te = train_test_split(np.arange(len(y_t)), test_size=test_size, random_state=seed)
            splits = [(tr, te)]

        for fold_idx, (idx_train, idx_test) in enumerate(splits):
            X_train, X_test = X_t[idx_train], X_t[idx_test]
            y_train, y_test = y_t[idx_train], y_t[idx_test]
            X_train, X_test = preprocess_train_test(X_train, X_test)
            fold_regressors = make_regressors(seed + fold_idx)
            for model_name, model in fold_regressors.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                metrics = {
                    "Embedding": method,
                    "FeatureFamily": family,
                    "AmineType": amine_label,
                    "Model": model_name,
                    "Target": target_name,
                    "Fold": int(fold_idx),
                    "SamplesTotal": int(len(y_t)),
                    "SamplesTrain": int(len(idx_train)),
                    "SamplesTest": int(len(idx_test)),
                    "R2": float(r2_score(y_test, pred)),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "MAE": float(mean_absolute_error(y_test, pred)),
                }
                results.append(metrics)

            if fold_idx != 0:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(y_test, pred, alpha=0.35, s=12)
            lo = min(y_test.min(), pred.min())
            hi = max(y_test.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            ax.set_xlabel(f"Actual {target_name}")
            ax.set_ylabel(f"Predicted {target_name}")
            ax.set_title(f"{method} | {family} | {model_name} | {target_name}")
            fig.tight_layout()
            safe_amine = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(amine_label))
            fname = f"scatter_{target_name}_{method}_{family}_{model_name}_{safe_amine}.png".replace(" ", "_")
            fig.savefig(os.path.join(out_dir, "scatter", fname), dpi=220)
            plt.close(fig)

            if amine_label == "ALL":
                amine_test = df.loc[valid, "amine_code"].values[idx_test]
                uniq = sorted(pd.unique(amine_test))
                cmap = plt.get_cmap("tab20", max(len(uniq), 1))
                color_map = {a: cmap(i % 20) for i, a in enumerate(uniq)}
                colors = [color_map[a] for a in amine_test]

                fig2, ax2 = plt.subplots(figsize=(8.2, 6.8))
                ax2.scatter(y_test, pred, c=colors, alpha=0.55, s=16, linewidths=0.0)
                ax2.plot([lo, hi], [lo, hi], "k--", linewidth=1)
                ax2.set_xlabel(f"Actual {target_name}")
                ax2.set_ylabel(f"Predicted {target_name}")
                ax2.set_title(f"{method} | {family} | {model_name} | {target_name} (colored by amine)")

                handles = []
                for a in uniq:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            label=str(a),
                            markerfacecolor=color_map[a],
                            markersize=6,
                        )
                    )
                if len(handles) <= 20:
                    ax2.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
                else:
                    # Keep legend readable when many amines are present.
                    counts = pd.Series(amine_test).value_counts()
                    top_amines = set(counts.head(20).index.tolist())
                    top_handles = [h for h in handles if h.get_label() in top_amines]
                    ax2.legend(
                        handles=top_handles,
                        title="Top 20 amines",
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=7,
                        frameon=False,
                    )

                fig2.tight_layout()
                fname2 = f"scatter_colored_{target_name}_{method}_{family}_{model_name}_{safe_amine}.png".replace(" ", "_")
                fig2.savefig(os.path.join(out_dir, "scatter", fname2), dpi=240)
                plt.close(fig2)

    return results


def plot_summary(df: pd.DataFrame, out_dir: str):
    for target in sorted(df["Target"].unique()):
        sub = df[df["Target"] == target]
        fig, ax = plt.subplots(figsize=(12, 6))
        sub = sub.copy()
        sub["MethodFamily"] = sub["Embedding"] + " | " + sub["FeatureFamily"]
        pivot = (
            sub.groupby(["MethodFamily", "Model"], as_index=False)["R2"]
            .mean()
            .sort_values("R2", ascending=False)
        )
        top = pivot.head(18)
        ax.bar(np.arange(len(top)), top["R2"].values)
        ax.set_xticks(np.arange(len(top)))
        ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(top["MethodFamily"], top["Model"])], rotation=65, ha="right")
        ax.set_title(f"{target}: R2 comparison (higher is better)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bar_{target}_r2.png"), dpi=240)
        plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "scatter"), exist_ok=True)

    log.info("Extracting parent-child functionalization pairs from KG...")
    pairs_df = get_parent_child_pairs(args.kg_path)
    if len(pairs_df) < args.min_samples:
        raise RuntimeError(f"Insufficient relation samples: {len(pairs_df)} < {args.min_samples}")
    log.info("Pairs available: %d", len(pairs_df))
    log.info("Amine types in pairs: %d", pairs_df["amine_code"].nunique())

    log.info("Loading chem properties and creating parent-level chem features...")
    df_prop = pd.read_csv(args.chem_csv)
    chem_df = build_chem_features(df_prop, pairs_df["parent_uri"].tolist())
    log.info("Chem features: %d MOFs x %d feats", chem_df.shape[0], chem_df.shape[1])

    emb_map: Dict[str, pd.DataFrame] = {
        "CompGCN": load_embedding_csv(args.compgcn_csv),
        "TransE": load_embedding_csv(args.transe_csv),
        "Node2Vec": load_node2vec_pt(args.node2vec_pt),
    }
    for k, v in emb_map.items():
        log.info("%s embeddings: %d MOFs x %d dims", k, v.shape[0], v.shape[1])

    all_results = []
    eval_slices = [("ALL", pairs_df)]
    requested_amines = {x.strip().lower() for x in args.amine_types.split(",") if x.strip()}
    amine_counts = pairs_df["amine_code"].value_counts()
    for amine_code, n in amine_counts.items():
        if requested_amines and amine_code not in requested_amines:
            continue
        if int(n) >= args.min_samples_per_amine:
            eval_slices.append((amine_code, pairs_df[pairs_df["amine_code"] == amine_code].copy()))
    for amine_label, slice_df in eval_slices:
        for method, emb_df in emb_map.items():
            for family in ["kg", "chem", "hybrid"]:
                res = evaluate_family_method(
                    pairs_df=slice_df,
                    method=method,
                    family=family,
                    emb_df=emb_df,
                    chem_df=chem_df,
                    test_size=args.test_size,
                    seed=args.seed,
                    out_dir=args.out_dir,
                    amine_label=amine_label,
                    cv_folds=args.cv_folds,
                )
                if res:
                    log.info("%s | %s | amine=%s -> %d metric rows", method, family, amine_label, len(res))
                    all_results.extend(res)

    if not all_results:
        raise RuntimeError("No relation-comparison results were generated.")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(args.out_dir, "relation_metrics_comparison.csv"), index=False)
    summary = (
        df_res.groupby(["Embedding", "FeatureFamily", "AmineType", "Model", "Target"], as_index=False)[["R2", "RMSE", "MAE"]]
        .mean()
        .sort_values(["Target", "AmineType", "R2"], ascending=[True, True, False])
    )
    summary.to_csv(os.path.join(args.out_dir, "relation_summary.csv"), index=False)
    plot_summary(df_res, args.out_dir)
    log.info("Saved outputs to %s", args.out_dir)


if __name__ == "__main__":
    main()
