#!/usr/bin/env python3
"""Reveal concept vectors from MOF embedding spaces using linear probes."""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from concepts.default_concepts import (
    CATEGORICAL_CONCEPT_COLUMNS,
    MULTIVALUE_TOKEN_COLUMNS,
    NUMERIC_CONCEPT_COLUMNS,
)

log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    method: str
    concept: str
    n_total: int
    n_pos: int
    n_neg: int
    accuracy: float
    f1: float
    roc_auc: float
    coef_norm: float
    split_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reveal concept vectors in MOF embeddings.")
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
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/concept_vectors))
    parser.add_argument("--methods", default="CompGCN,Node2Vec,TransE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of seeds; probes run once per seed to produce per-seed AUCs for error bars",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--high_quantile", type=float, default=0.75)
    parser.add_argument("--low_quantile", type=float, default=0.25)
    parser.add_argument("--min_class_size", type=int, default=80)
    parser.add_argument("--min_total", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--max_category_levels", type=int, default=8)
    parser.add_argument("--max_token_levels", type=int, default=10)
    parser.add_argument("--max_concepts_for_plot", type=int, default=8)
    parser.add_argument("--projection_sample_n", type=int, default=6000)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())


def _ensure_mof_uri_index(df: pd.DataFrame) -> pd.DataFrame:
    if "mof_uri" in df.columns:
        return df.set_index("mof_uri")
    if "uri" in df.columns:
        return df.set_index("uri")
    first = df.columns[0]
    if df[first].dtype == object:
        return df.set_index(first)
    raise ValueError("Could not identify URI column in embedding file.")


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
        frag = uri.split("#")[-1] if "#" in uri else uri
        if not (frag.startswith("MOF_") or frag.startswith("FuncMOF_")):
            continue
        rows.append([uri] + emb[idx].tolist())
    cols = ["mof_uri"] + [f"emb_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(rows, columns=cols).set_index("mof_uri")


def load_embeddings(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    raw = {
        "CompGCN": load_embedding_csv(args.compgcn_csv),
        "Node2Vec": load_node2vec_pt(args.node2vec_pt),
        "TransE": load_embedding_csv(args.transe_csv),
    }
    wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
    selected = {name: df for name, df in raw.items() if name in wanted}
    if not selected:
        raise ValueError(f"No valid methods selected from --methods={args.methods}")
    for name, df in selected.items():
        log.info("%s embeddings loaded: %d MOFs x %d dims", name, df.shape[0], df.shape[1])
    return selected


def _build_numeric_concepts(
    df: pd.DataFrame,
    high_q: float,
    low_q: float,
    min_class_size: int,
) -> Dict[str, pd.Series]:
    concepts: Dict[str, pd.Series] = {}
    for col in NUMERIC_CONCEPT_COLUMNS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        valid = s.dropna()
        if valid.empty:
            continue
        hi = float(valid.quantile(high_q))
        lo = float(valid.quantile(low_q))
        if hi <= lo:
            continue
        c_hi = pd.Series(np.nan, index=df.index, dtype="float64")
        c_lo = pd.Series(np.nan, index=df.index, dtype="float64")
        c_hi.loc[valid.index] = (valid >= hi).astype(float)
        c_lo.loc[valid.index] = (valid <= lo).astype(float)
        if int((c_hi == 1).sum()) >= min_class_size and int((c_hi == 0).sum()) >= min_class_size:
            concepts[f"high_{_safe(col)}"] = c_hi
        if int((c_lo == 1).sum()) >= min_class_size and int((c_lo == 0).sum()) >= min_class_size:
            concepts[f"low_{_safe(col)}"] = c_lo
    return concepts


def _build_categorical_concepts(
    df: pd.DataFrame,
    min_class_size: int,
    max_levels: int,
) -> Dict[str, pd.Series]:
    concepts: Dict[str, pd.Series] = {}
    for col in CATEGORICAL_CONCEPT_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col].fillna("").astype(str).str.strip()
        counts = s[s != ""].value_counts().head(max_levels)
        for level in counts.index:
            c = pd.Series(np.nan, index=df.index, dtype="float64")
            mask = s != ""
            c.loc[mask] = (s.loc[mask] == level).astype(float)
            if int((c == 1).sum()) >= min_class_size and int((c == 0).sum()) >= min_class_size:
                concepts[f"{_safe(col)}__is__{_safe(level)}"] = c
    return concepts


def _build_multivalue_token_concepts(
    df: pd.DataFrame,
    min_class_size: int,
    max_levels: int,
) -> Dict[str, pd.Series]:
    concepts: Dict[str, pd.Series] = {}
    for col in MULTIVALUE_TOKEN_COLUMNS:
        if col not in df.columns:
            continue
        tokens: List[str] = []
        series = df[col].fillna("").astype(str)
        for raw in series:
            parts = [p.strip() for p in raw.split(";") if p.strip()]
            tokens.extend(parts)
        if not tokens:
            continue
        top_tokens = pd.Series(tokens).value_counts().head(max_levels).index.tolist()
        for token in top_tokens:
            c = pd.Series(np.nan, index=df.index, dtype="float64")
            nonempty = series != ""
            c.loc[nonempty] = series.loc[nonempty].map(
                lambda x: float(token in [p.strip() for p in x.split(";") if p.strip()])
            )
            if int((c == 1).sum()) >= min_class_size and int((c == 0).sum()) >= min_class_size:
                concepts[f"{_safe(col)}__has__{_safe(token)}"] = c
    return concepts


def build_auto_concepts(args: argparse.Namespace, chem_df: pd.DataFrame) -> pd.DataFrame:
    base = chem_df.copy()
    if "mof_uri" not in base.columns:
        raise ValueError("Expected column 'mof_uri' in chemical properties CSV.")
    base = base.drop_duplicates(subset=["mof_uri"]).set_index("mof_uri")

    concepts: Dict[str, pd.Series] = {}
    concepts.update(_build_numeric_concepts(base, args.high_quantile, args.low_quantile, args.min_class_size))
    concepts.update(_build_categorical_concepts(base, args.min_class_size, args.max_category_levels))
    concepts.update(_build_multivalue_token_concepts(base, args.min_class_size, args.max_token_levels))

    if not concepts:
        raise RuntimeError("No valid concepts were generated from chemical properties.")

    concept_df = pd.DataFrame(concepts, index=base.index)
    keep_cols = []
    for col in concept_df.columns:
        y = concept_df[col].dropna()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos >= args.min_class_size and n_neg >= args.min_class_size and (n_pos + n_neg) >= args.min_total:
            keep_cols.append(col)
    concept_df = concept_df[keep_cols]
    if concept_df.empty:
        raise RuntimeError("All generated concepts were filtered out by class-size constraints.")
    log.info("Generated %d concepts after filtering.", concept_df.shape[1])
    return concept_df


def train_probes(
    args: argparse.Namespace,
    embeddings: Dict[str, pd.DataFrame],
    concepts: pd.DataFrame,
    out_dir: str,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, StandardScaler]]]:
    metrics_rows: List[dict] = []
    vectors_by_method: Dict[str, pd.DataFrame] = {}
    vector_map: Dict[str, Dict[str, np.ndarray]] = {}
    scaler_map: Dict[str, Dict[str, StandardScaler]] = {}

    for method, emb_df in embeddings.items():
        rows = []
        method_vectors: Dict[str, np.ndarray] = {}
        method_scalers: Dict[str, StandardScaler] = {}
        for concept in concepts.columns:
            y_all = concepts[concept]
            common = emb_df.index.intersection(y_all.index)
            if len(common) < args.min_total:
                continue
            y = y_all.loc[common].dropna()
            if y.empty:
                continue
            x_df = emb_df.loc[y.index]
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos < args.min_class_size or n_neg < args.min_class_size:
                continue

            x = x_df.values.astype(np.float32)
            y_np = y.values.astype(np.int64)

            seeds_to_run = args.seeds if args.seeds else [args.seed]
            canonical_seed = seeds_to_run[0]
            canonical_coef = None
            canonical_scaler = None

            for s_idx, s in enumerate(seeds_to_run):
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y_np, test_size=args.test_size, random_state=s, stratify=y_np
                )

                scaler = StandardScaler()
                x_train_s = scaler.fit_transform(x_train)
                x_test_s = scaler.transform(x_test)

                probe = LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=s,
                )
                probe.fit(x_train_s, y_train)
                y_pred = probe.predict(x_test_s)
                y_prob = probe.predict_proba(x_test_s)[:, 1]

                try:
                    auc = float(roc_auc_score(y_test, y_prob))
                except Exception:
                    auc = float("nan")
                acc = float(accuracy_score(y_test, y_pred))
                f1 = float(f1_score(y_test, y_pred))

                coef = probe.coef_[0].astype(np.float64)
                coef_norm = float(np.linalg.norm(coef))
                if coef_norm > 0:
                    coef = coef / coef_norm

                metrics_rows.append(
                    ProbeResult(
                        method=method,
                        concept=concept,
                        n_total=int(len(y_np)),
                        n_pos=n_pos,
                        n_neg=n_neg,
                        accuracy=acc,
                        f1=f1,
                        roc_auc=auc,
                        coef_norm=coef_norm,
                        split_seed=s,
                    ).__dict__
                )
                if s == canonical_seed:
                    canonical_coef = coef
                    canonical_scaler = scaler

            # Downstream analyses (similarity, projections) use the canonical-seed coef
            rows.append({"concept": concept, **{f"w_{i}": float(v) for i, v in enumerate(canonical_coef)}})
            method_vectors[concept] = canonical_coef
            method_scalers[concept] = canonical_scaler

        vectors_df = pd.DataFrame(rows)
        vectors_by_method[method] = vectors_df
        vector_map[method] = method_vectors
        scaler_map[method] = method_scalers
        if not vectors_df.empty:
            vectors_df.to_csv(os.path.join(out_dir, f"concept_vectors_{method}.csv"), index=False)
        log.info("%s: trained %d concept probes.", method, len(rows))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["method", "roc_auc", "f1"], ascending=[True, False, False])
    metrics_df.to_csv(os.path.join(out_dir, "concept_probe_metrics.csv"), index=False)
    return metrics_df, vectors_by_method, vector_map, scaler_map


def compute_similarity_tables(
    out_dir: str,
    vector_map: Dict[str, Dict[str, np.ndarray]],
) -> None:
    for method, concept_vectors in vector_map.items():
        if not concept_vectors:
            continue
        names = sorted(concept_vectors.keys())
        mat = np.zeros((len(names), len(names)), dtype=np.float64)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                va, vb = concept_vectors[a], concept_vectors[b]
                denom = np.linalg.norm(va) * np.linalg.norm(vb)
                mat[i, j] = float(np.dot(va, vb) / denom) if denom > 0 else 0.0
        sim_df = pd.DataFrame(mat, index=names, columns=names)
        sim_df.to_csv(os.path.join(out_dir, f"concept_similarity_{method}.csv"))

    rows = []
    methods = sorted(vector_map.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            common = sorted(set(vector_map[m1].keys()).intersection(vector_map[m2].keys()))
            for concept in common:
                v1 = vector_map[m1][concept]
                v2 = vector_map[m2][concept]
                if v1.shape != v2.shape:
                    continue
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                cos = float(np.dot(v1, v2) / denom) if denom > 0 else 0.0
                rows.append({"method_a": m1, "method_b": m2, "concept": concept, "cosine": cos})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "concept_similarity_cross_method.csv"), index=False)


def export_alignment_rankings(
    args: argparse.Namespace,
    out_dir: str,
    embeddings: Dict[str, pd.DataFrame],
    vector_map: Dict[str, Dict[str, np.ndarray]],
    scaler_map: Dict[str, Dict[str, StandardScaler]],
) -> None:
    rank_dir = os.path.join(out_dir, "rankings")
    os.makedirs(rank_dir, exist_ok=True)

    for method, emb_df in embeddings.items():
        for concept, vec in vector_map.get(method, {}).items():
            scaler = scaler_map[method][concept]
            x = scaler.transform(emb_df.values.astype(np.float32))
            vec_norm = np.linalg.norm(vec)
            x_norm = np.linalg.norm(x, axis=1)
            denom = x_norm * vec_norm + 1e-12
            cosine = (x @ vec) / denom
            margin = x @ vec

            scored = pd.DataFrame(
                {
                    "mof_uri": emb_df.index.values,
                    "cosine_alignment": cosine,
                    "signed_margin": margin,
                }
            ).sort_values("cosine_alignment", ascending=False)

            top = scored.head(args.top_k)
            bottom = scored.tail(args.top_k).sort_values("cosine_alignment", ascending=True)
            top.to_csv(os.path.join(rank_dir, f"concept_alignment_topk_{method}_{_safe(concept)}.csv"), index=False)
            bottom.to_csv(
                os.path.join(rank_dir, f"concept_alignment_bottomk_{method}_{_safe(concept)}.csv"), index=False
            )


def _project_embeddings_2d(x: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=seed)  # type: ignore
        y = reducer.fit_transform(x)
        return y, "UMAP"
    x_pca = PCA(n_components=min(50, x.shape[1], max(2, x.shape[0] - 1)), random_state=seed).fit_transform(x)
    y = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(x_pca)
    return y, "t-SNE"


def visualize_concepts(
    args: argparse.Namespace,
    out_dir: str,
    embeddings: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    vector_map: Dict[str, Dict[str, np.ndarray]],
) -> None:
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for method, emb_df in embeddings.items():
        if method not in vector_map or not vector_map[method]:
            continue

        x = emb_df.values.astype(np.float32)
        if x.shape[0] > args.projection_sample_n:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(np.arange(x.shape[0]), size=args.projection_sample_n, replace=False)
            x = x[idx]
            uris = emb_df.index.values[idx]
        else:
            uris = emb_df.index.values

        scaler = StandardScaler()
        x_s = scaler.fit_transform(x)
        y2d, proj_name = _project_embeddings_2d(x_s, args.seed)

        # Linear surrogate maps standardized high-dim vectors to 2D coordinates.
        # This allows direction overlays on top of nonlinear projections.
        map2d = LinearRegression(fit_intercept=False)
        map2d.fit(x_s, y2d)
        coef_2d = map2d.coef_.T  # [D,2]

        top_concepts = (
            metrics_df[metrics_df["method"] == method]
            .sort_values("roc_auc", ascending=False)["concept"]
            .head(args.max_concepts_for_plot)
            .tolist()
        )
        if not top_concepts:
            continue

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(y2d[:, 0], y2d[:, 1], s=8, alpha=0.35, linewidths=0)
        ax.set_title(f"{method} projection ({proj_name})")
        ax.set_xlabel(f"{proj_name} 1")
        ax.set_ylabel(f"{proj_name} 2")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"projection_{method}.png"), dpi=260)
        plt.close(fig)

        center = np.mean(y2d, axis=0)
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.scatter(y2d[:, 0], y2d[:, 1], s=7, alpha=0.28, linewidths=0, color="gray")

        cmap = plt.get_cmap("tab10", max(1, len(top_concepts)))
        for i, concept in enumerate(top_concepts):
            vec = vector_map[method][concept].astype(np.float64)
            v2d = vec @ coef_2d
            norm = np.linalg.norm(v2d)
            if norm > 0:
                v2d = v2d / norm
            scale = np.percentile(np.linalg.norm(y2d - center, axis=1), 75)
            arrow = v2d * scale
            ax2.arrow(
                center[0],
                center[1],
                arrow[0],
                arrow[1],
                width=0.01,
                alpha=0.9,
                color=cmap(i),
                length_includes_head=True,
            )
            ax2.text(center[0] + arrow[0], center[1] + arrow[1], concept, fontsize=8, color=cmap(i))

        ax2.set_title(f"{method}: projected concept directions ({proj_name})")
        ax2.set_xlabel(f"{proj_name} 1")
        ax2.set_ylabel(f"{proj_name} 2")
        fig2.tight_layout()
        fig2.savefig(os.path.join(plot_dir, f"projection_with_concepts_{method}.png"), dpi=260)
        plt.close(fig2)

        pd.DataFrame({"mof_uri": uris, "x": y2d[:, 0], "y": y2d[:, 1]}).to_csv(
            os.path.join(out_dir, f"projection_points_{method}.csv"), index=False
        )


def write_summary(
    args: argparse.Namespace,
    out_dir: str,
    concepts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Concept Vector Run Summary")
    lines.append("")
    lines.append(f"- Methods: {args.methods}")
    lines.append(f"- Concepts generated: {concepts_df.shape[1]}")
    lines.append(f"- Probe rows: {len(metrics_df)}")
    lines.append(f"- Seed: {args.seed}")
    lines.append("")
    if metrics_df.empty:
        lines.append("No valid probe fits were produced.")
    else:
        lines.append("## Best concepts by method (AUC)")
        for method in sorted(metrics_df["method"].unique()):
            sub = metrics_df[metrics_df["method"] == method].sort_values("roc_auc", ascending=False).head(8)
            lines.append("")
            lines.append(f"### {method}")
            for _, row in sub.iterrows():
                lines.append(
                    f"- {row['concept']}: AUC={row['roc_auc']:.3f}, ACC={row['accuracy']:.3f}, F1={row['f1']:.3f}"
                )
    with open(os.path.join(out_dir, "concept_run_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    chem_df = pd.read_csv(args.chem_csv)
    embeddings = load_embeddings(args)
    concepts_df = build_auto_concepts(args, chem_df)

    metrics_df, _, vector_map, scaler_map = train_probes(args, embeddings, concepts_df, args.out_dir)
    compute_similarity_tables(args.out_dir, vector_map)
    export_alignment_rankings(args, args.out_dir, embeddings, vector_map, scaler_map)
    visualize_concepts(args, args.out_dir, embeddings, metrics_df, vector_map)
    write_summary(args, args.out_dir, concepts_df, metrics_df)
    log.info("Concept-vector pipeline complete. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()

