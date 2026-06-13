#!/usr/bin/env python3
"""Family-aware link prediction stratified by predicate.

For each requested embedding (CompGCN / TransE / Node2Vec) and each relation
present in the MOF subgraph, computes AUC / Hits@10 under the same family-aware
split used in ``eval_link_prediction_family.py``. Writes
``family_aware_relation_quality.csv`` for figure generation.
"""

import argparse
import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from eval_link_prediction_family import (
    SCORER_BY_METHOD,
    evaluate_link_prediction,
    load_embeddings,
    load_node2vec_pt,
    parse_kg_edges_fast,
    split_edges_by_family,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Per-relation family-aware LP")
    p.add_argument("--kg_path", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl"))
    p.add_argument("--compgcn_csv", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_compgcn_embeddings_256d_3layers.csv))
    p.add_argument("--transe_csv", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings"/mof_transe_embeddings_256d.csv))
    p.add_argument("--node2vec_pt", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_256d_p1.0_q1.0.pt))
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/link_prediction_family_eval))
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--n_negative", type=int, default=50)
    p.add_argument("--min_edges", type=int, default=50,
                   help="Skip relations with fewer than this many total edges")
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--methods", type=str, default="CompGCN,TransE,Node2Vec")
    return p.parse_args()


def edges_by_relation(edges: List[tuple]) -> dict:
    grouped = defaultdict(list)
    for s, p, o in edges:
        grouped[p].append((s, p, o))
    return grouped


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    method_paths = {
        "CompGCN": args.compgcn_csv,
        "TransE": args.transe_csv,
        "Node2Vec": args.node2vec_pt,
    }

    rows = []
    for method in methods:
        path = method_paths.get(method)
        if not path or not os.path.exists(path):
            log.warning(f"No embeddings for {method} at {path}")
            continue

        log.info(f"\n===== {method} =====")
        if method == "Node2Vec":
            embeddings = load_node2vec_pt(path)
        else:
            embeddings = load_embeddings(path)

        edges = parse_kg_edges_fast(args.kg_path, embeddings)
        if len(edges) < args.min_edges:
            log.warning(f"Too few edges ({len(edges)}) for {method}")
            continue
        use_cosine = SCORER_BY_METHOD.get(method, False)

        rel_groups = edges_by_relation(edges)
        log.info(f"{method}: {len(rel_groups)} relations, {len(edges)} edges total")

        for rel, rel_edges in sorted(rel_groups.items()):
            if len(rel_edges) < args.min_edges:
                log.info(f"  skip {rel}: only {len(rel_edges)} edges")
                continue
            for seed in args.seeds:
                train, test, _ = split_edges_by_family(rel_edges, args.test_ratio, seed)
                if len(test) < 5:
                    continue
                metrics = evaluate_link_prediction(
                    test, train, embeddings,
                    n_negative=args.n_negative, seed=seed,
                    use_cosine=use_cosine,
                )
                rows.append({
                    "method": method,
                    "relation": rel,
                    "seed": seed,
                    "n_edges_total": len(rel_edges),
                    "AUC": metrics["AUC"],
                    "Hits@10": metrics["Hits@10"],
                    "n_test_edges": metrics["n_test_edges"],
                })
                log.info(f"  {rel} seed={seed}: AUC={metrics['AUC']:.4f} Hits@10={metrics['Hits@10']:.4f}")

    if not rows:
        log.error("No rows produced.")
        return

    df = pd.DataFrame(rows)
    out = os.path.join(args.out_dir, "family_aware_relation_quality.csv")
    df.to_csv(out, index=False)
    log.info(f"Wrote {out} ({len(df)} rows)")

    # Aggregate mean/std per (method, relation)
    agg = (
        df.groupby(["method", "relation"])
        .agg(
            AUC_mean=("AUC", "mean"),
            AUC_std=("AUC", lambda s: s.std(ddof=0) if len(s) > 1 else 0.0),
            Hits10_mean=("Hits@10", "mean"),
            Hits10_std=("Hits@10", lambda s: s.std(ddof=0) if len(s) > 1 else 0.0),
            n_seeds=("seed", "nunique"),
            n_edges_total=("n_edges_total", "max"),
        )
        .reset_index()
    )
    agg_out = os.path.join(args.out_dir, "family_aware_relation_quality_summary.csv")
    agg.to_csv(agg_out, index=False)
    log.info(f"Wrote {agg_out}")


if __name__ == "__main__":
    main()
