#!/usr/bin/env python3
"""
Re-evaluate link prediction with family-aware edge splits.
Uses pre-trained embeddings but holds out edges by MOF family.
"""

import argparse
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Family-aware link prediction evaluation")
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
        help="PyTorch checkpoint with {'embeddings': Tensor[N,256], 'ent2id': {uri: idx}}",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/link_prediction_family_eval),
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of seeds (overrides --seed) to compute mean±SD across seeds",
    )
    parser.add_argument("--n_negative", type=int, default=50, help="Negative samples per positive edge")
    parser.add_argument(
        "--methods",
        type=str,
        default="CompGCN,TransE,Node2Vec",
        help="Comma-separated list of methods to evaluate",
    )
    return parser.parse_args()


def get_mof_family(uri: str) -> str:
    """Extract MOF family from URI, grouping parent and functionalized variants."""
    if not uri or not isinstance(uri, str):
        return "unknown"

    # Extract fragment after hash
    frag = uri.split("#")[-1] if "#" in uri else uri

    if frag.startswith("FuncMOF_"):
        # FuncMOF_HKUST1_deen -> HKUST1
        parts = frag.replace("FuncMOF_", "").split("_")
        return parts[0] if parts else frag
    elif frag.startswith("MOF_"):
        # MOF_HKUST1 -> HKUST1
        return frag.replace("MOF_", "").split("_")[0]
    else:
        return frag.split("_")[0]


def load_node2vec_pt(path: str) -> Dict[str, np.ndarray]:
    """Load Node2Vec embeddings from a PyTorch .pt checkpoint.

    Expected format: dict with 'embeddings' (Tensor[N,D]) and 'ent2id' (URI -> int).
    Returns a dict keyed by both the full URI and its fragment (after '#').
    """
    import torch

    log.info(f"Loading Node2Vec embeddings from {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    emb_tensor = ckpt["embeddings"]
    ent2id = ckpt["ent2id"]

    if hasattr(emb_tensor, "detach"):
        emb_np = emb_tensor.detach().cpu().numpy().astype(np.float32)
    else:
        emb_np = np.asarray(emb_tensor, dtype=np.float32)

    embeddings: Dict[str, np.ndarray] = {}
    for uri, idx in ent2id.items():
        vec = emb_np[idx]
        embeddings[uri] = vec
        frag = uri.split("#")[-1] if "#" in uri else uri
        embeddings[frag] = vec
    log.info(
        f"Loaded {len(embeddings)} embeddings (with fragment keys) with dim={emb_np.shape[1]}"
    )
    return embeddings


def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """Load embeddings from CSV into dict."""
    log.info(f"Loading embeddings from {path}")
    df = pd.read_csv(path)

    # Find URI column
    uri_col = None
    for col in ["mof_uri", "uri", df.columns[0]]:
        if col in df.columns:
            uri_col = col
            break

    if uri_col is None:
        raise ValueError("Could not find URI column")

    # Get embedding columns
    emb_cols = [c for c in df.columns if c.startswith("emb_") or c != uri_col]
    emb_cols = [c for c in emb_cols if c != uri_col]

    # Filter to numeric columns only
    numeric_cols = df[emb_cols].select_dtypes(include=[np.number]).columns.tolist()

    embeddings = {}
    for _, row in df.iterrows():
        uri = str(row[uri_col])
        emb = row[numeric_cols].values.astype(np.float32)
        # Store by both full URI and fragment (after #)
        embeddings[uri] = emb
        frag = uri.split('#')[-1] if '#' in uri else uri
        embeddings[frag] = emb

    log.info(f"Loaded {len(embeddings)} embeddings (with fragment keys) with dim={len(numeric_cols)}")
    return embeddings


def extract_mof_edges(kg_path: str) -> List[Tuple[str, str, str]]:
    """Extract edges involving MOFs from KG (without loading full graph)."""
    log.info(f"Extracting MOF edges from {kg_path}")

    mof_prefix = "http://emmo.info/domain-mof/mof-ontology#"
    edges = []

    # Relations to include (MOF-relevant)
    include_relations = {
        "hasLinker", "hasMetalNode", "hasTopology", "hasSBU",
        "hasStructuralProperty", "hasComputationalProperty", "hasPhysicalProperty",
        "hasSynthesisMethod", "hasCapability", "isFunctionalizedFrom",
        "hasFormula", "hasMOFid"
    }

    with open(kg_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('@') or line.startswith('#'):
                continue

            # Simple TTL parsing for subject predicate object patterns
            # Look for MOF entities
            if "MOF_" in line or "FuncMOF_" in line:
                parts = line.split()
                if len(parts) >= 3:
                    subj = parts[0].strip('<>').replace(mof_prefix, '')
                    pred = parts[1].strip('<>').replace(mof_prefix, '')
                    obj = parts[2].strip('<>.').replace(mof_prefix, '')

                    # Filter to relevant relations
                    pred_name = pred.split('#')[-1] if '#' in pred else pred
                    if any(rel in pred_name for rel in include_relations):
                        if "MOF_" in subj or "FuncMOF_" in subj:
                            edges.append((subj, pred_name, obj))

    log.info(f"Extracted {len(edges)} MOF edges")
    return edges


def parse_kg_edges_fast(kg_path: str, embeddings: Dict[str, np.ndarray]) -> List[Tuple[str, str, str]]:
    """Fast extraction of edges between entities that have embeddings."""
    log.info(f"Parsing edges from {kg_path} (fast mode)")

    # Get all URIs with embeddings
    emb_uris = set(embeddings.keys())
    # Also match without prefix - extract fragment after #
    emb_frags = {uri.split('#')[-1] if '#' in uri else uri for uri in emb_uris}

    edges = []
    current_subj = None

    with open(kg_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                log.info(f"  Processed {i/1e6:.1f}M lines, found {len(edges)} edges...")

            raw_line = line
            line = line.strip()
            if not line or line.startswith('@') or line.startswith('#'):
                continue

            # Check if this is a continuation line (starts with whitespace)
            is_continuation = raw_line[0] in (' ', '\t') if raw_line else False

            if is_continuation and current_subj:
                # Continuation: parse predicate object
                parts = line.rstrip(' .;,').split(None, 1)
                if len(parts) >= 2:
                    pred = parts[0].strip('<>')
                    obj = parts[1].strip('<>').rstrip(' .;,')
                    subj_frag = current_subj
                else:
                    continue
            else:
                # New subject line
                parts = line.split(None, 2)
                if len(parts) < 2:
                    continue

                subj = parts[0].strip('<>')
                subj_frag = subj.split('#')[-1] if '#' in subj else subj.lstrip(':')

                # Update current subject (ends with . means end of block)
                if not line.rstrip().endswith('.'):
                    current_subj = subj_frag
                else:
                    current_subj = None

                if len(parts) < 3:
                    continue

                pred = parts[1].strip('<>')
                obj = parts[2].rstrip(' .;,').strip('<>')

            # Normalize predicate and object
            pred_frag = pred.split('#')[-1] if '#' in pred else pred.lstrip(':')
            obj_frag = obj.split('#')[-1] if '#' in obj else obj.lstrip(':')

            # For MOF-to-MOF edges (both endpoints must be MOFs)
            is_subj_mof = "MOF_" in subj_frag
            is_obj_mof = "MOF_" in obj_frag

            if not (is_subj_mof and is_obj_mof):
                continue

            # Check if both endpoints have embeddings
            subj_match = subj_frag in emb_frags
            obj_match = obj_frag in emb_frags

            if subj_match and obj_match:
                edges.append((subj_frag, pred_frag, obj_frag))

    log.info(f"Found {len(edges)} edges between embedded entities")
    return edges


def split_edges_by_family(
    edges: List[Tuple[str, str, str]],
    test_ratio: float,
    seed: int
) -> Tuple[List[Tuple], List[Tuple], Dict[str, str]]:
    """Split edges by MOF family - all edges for test families go to test set."""

    # Get all MOF URIs and their families
    mof_to_family = {}
    for subj, pred, obj in edges:
        if "MOF_" in subj or "FuncMOF_" in subj:
            mof_to_family[subj] = get_mof_family(subj)
        if "MOF_" in obj or "FuncMOF_" in obj:
            mof_to_family[obj] = get_mof_family(obj)

    # Get unique families
    families = list(set(mof_to_family.values()))
    log.info(f"Found {len(families)} unique MOF families")

    # Split families
    np.random.seed(seed)
    np.random.shuffle(families)
    n_test = int(len(families) * test_ratio)
    test_families = set(families[:n_test])
    train_families = set(families[n_test:])

    log.info(f"Test families: {n_test}, Train families: {len(train_families)}")

    # Split edges based on family membership
    train_edges = []
    test_edges = []

    for edge in edges:
        subj, pred, obj = edge
        subj_fam = mof_to_family.get(subj, "unknown")
        obj_fam = mof_to_family.get(obj, "unknown")

        # Edge goes to test if EITHER endpoint is in test family
        if subj_fam in test_families or obj_fam in test_families:
            test_edges.append(edge)
        else:
            train_edges.append(edge)

    log.info(f"Train edges: {len(train_edges)}, Test edges: {len(test_edges)}")
    return train_edges, test_edges, mof_to_family


def score_edge_transe(
    head_emb: np.ndarray,
    rel_emb: np.ndarray,
    tail_emb: np.ndarray
) -> float:
    """TransE score: -||h + r - t||"""
    return -np.linalg.norm(head_emb + rel_emb - tail_emb)


def score_edge_dot(head_emb: np.ndarray, tail_emb: np.ndarray) -> float:
    """Simple dot product score."""
    return np.dot(head_emb, tail_emb)


def score_edge_cosine(head_emb: np.ndarray, tail_emb: np.ndarray) -> float:
    """Cosine similarity score (for ReLU-activated embeddings like CompGCN)."""
    norm_h = np.linalg.norm(head_emb)
    norm_t = np.linalg.norm(tail_emb)
    if norm_h < 1e-8 or norm_t < 1e-8:
        return 0.0
    return np.dot(head_emb, tail_emb) / (norm_h * norm_t)


def score_edge_l2(head_emb: np.ndarray, tail_emb: np.ndarray) -> float:
    """Negative L2 distance score (closer = higher score)."""
    return -np.linalg.norm(head_emb - tail_emb)


def score_edge_neg_cosine(head_emb: np.ndarray, tail_emb: np.ndarray) -> float:
    """Negative cosine similarity (for embeddings where related nodes are farther apart)."""
    norm_h = np.linalg.norm(head_emb)
    norm_t = np.linalg.norm(tail_emb)
    if norm_h < 1e-8 or norm_t < 1e-8:
        return 0.0
    return -np.dot(head_emb, tail_emb) / (norm_h * norm_t)


def evaluate_link_prediction(
    test_edges: List[Tuple[str, str, str]],
    train_edges: List[Tuple[str, str, str]],
    embeddings: Dict[str, np.ndarray],
    n_negative: int = 50,
    seed: int = 42,
    use_cosine: bool = False
) -> Dict[str, float]:
    """Evaluate link prediction on test edges."""

    np.random.seed(seed)

    # Choose scoring function based on embedding type
    if use_cosine == "l2":
        score_fn = score_edge_l2
    elif use_cosine == "neg_cosine":
        score_fn = score_edge_neg_cosine
    elif use_cosine:
        score_fn = score_edge_cosine
    else:
        score_fn = score_edge_dot

    # Get all entities as numpy array for fast sampling
    all_entities = np.array(list(embeddings.keys()))
    n_entities = len(all_entities)

    # Build set of true edges for filtering
    true_edges = set((s, o) for s, p, o in train_edges + test_edges)

    scores_pos = []
    scores_neg = []

    log.info(f"Evaluating {len(test_edges)} test edges (cosine={use_cosine})...")

    for i, (subj, pred, obj) in enumerate(test_edges):
        if i % 100 == 0 and i > 0:
            log.info(f"  Evaluated {i}/{len(test_edges)} edges...")

        # Get embeddings
        subj_emb = embeddings.get(subj)
        obj_emb = embeddings.get(obj)

        if subj_emb is None or obj_emb is None:
            continue

        # Positive score
        pos_score = score_fn(subj_emb, obj_emb)
        scores_pos.append(pos_score)

        # Negative samples (corrupt tail) - batch sample for speed
        neg_indices = np.random.randint(0, n_entities, size=n_negative * 3)
        neg_count = 0
        for idx in neg_indices:
            if neg_count >= n_negative:
                break
            neg_tail = all_entities[idx]
            if (subj, neg_tail) not in true_edges:
                neg_emb = embeddings.get(neg_tail)
                if neg_emb is not None:
                    neg_score = score_fn(subj_emb, neg_emb)
                    scores_neg.append(neg_score)
                    neg_count += 1

    # Compute metrics
    y_true = [1] * len(scores_pos) + [0] * len(scores_neg)
    y_score = scores_pos + scores_neg

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    # Hits@K
    all_scores = np.array(scores_pos + scores_neg)
    all_labels = np.array(y_true)

    # For each positive, count how many negatives rank higher
    hits_at_1 = 0
    hits_at_10 = 0
    hits_at_50 = 0

    n_pos = len(scores_pos)
    for i, pos_score in enumerate(scores_pos):
        # Get corresponding negatives (n_negative per positive)
        start_neg = i * n_negative
        end_neg = start_neg + n_negative
        if end_neg > len(scores_neg):
            continue
        neg_scores = scores_neg[start_neg:end_neg]

        rank = 1 + sum(1 for ns in neg_scores if ns > pos_score)
        if rank <= 1:
            hits_at_1 += 1
        if rank <= 10:
            hits_at_10 += 1
        if rank <= 50:
            hits_at_50 += 1

    metrics = {
        "AUC": auc,
        "AP": ap,
        "Hits@1": hits_at_1 / n_pos if n_pos > 0 else 0,
        "Hits@10": hits_at_10 / n_pos if n_pos > 0 else 0,
        "Hits@50": hits_at_50 / n_pos if n_pos > 0 else 0,
        "n_test_edges": len(scores_pos),
        "n_negative_per_edge": n_negative,
    }

    return metrics


SCORER_BY_METHOD = {
    "CompGCN": "neg_cosine",
    "TransE": False,          # dot product
    "Node2Vec": True,         # cosine similarity
}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    methods_requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    method_paths = {
        "CompGCN": args.compgcn_csv,
        "TransE": args.transe_csv,
        "Node2Vec": args.node2vec_pt,
    }

    seeds = args.seeds if args.seeds else [args.seed]
    log.info(f"Running methods={methods_requested} across seeds={seeds}")

    results = []

    for method in methods_requested:
        emb_path = method_paths.get(method)
        if not emb_path or not os.path.exists(emb_path):
            log.warning(f"Embedding file not found for {method}: {emb_path}")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Evaluating {method}")
        log.info(f"{'='*60}")

        if method == "Node2Vec":
            embeddings = load_node2vec_pt(emb_path)
        else:
            embeddings = load_embeddings(emb_path)

        edges = parse_kg_edges_fast(args.kg_path, embeddings)

        if len(edges) < 100:
            log.warning(f"Too few edges found ({len(edges)}), skipping {method}")
            continue

        use_cosine = SCORER_BY_METHOD.get(method, False)

        for seed in seeds:
            train_edges, test_edges, _ = split_edges_by_family(
                edges, args.test_ratio, seed
            )

            if len(test_edges) < 10:
                log.warning(f"Too few test edges ({len(test_edges)}) for {method} seed={seed}")
                continue

            metrics = evaluate_link_prediction(
                test_edges, train_edges, embeddings,
                n_negative=args.n_negative, seed=seed,
                use_cosine=use_cosine,
            )
            metrics["method"] = method
            metrics["seed"] = seed
            results.append(metrics)

            log.info(f"{method} seed={seed}: AUC={metrics['AUC']:.4f}, Hits@10={metrics['Hits@10']:.4f}")

    if results:
        df = pd.DataFrame(results)
        out_path = os.path.join(args.out_dir, "family_aware_link_prediction.csv")
        df.to_csv(out_path, index=False)
        log.info(f"\nResults saved to {out_path}")

        summary_rows = []
        for m, grp in df.groupby("method"):
            summary_rows.append({
                "method": m,
                "AUC_mean": grp["AUC"].mean(),
                "AUC_std": grp["AUC"].std(ddof=0) if len(grp) > 1 else 0.0,
                "Hits@10_mean": grp["Hits@10"].mean(),
                "Hits@10_std": grp["Hits@10"].std(ddof=0) if len(grp) > 1 else 0.0,
                "n_seeds": len(grp),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            os.path.join(args.out_dir, "family_aware_link_prediction_summary.csv"),
            index=False,
        )

        summary_path = os.path.join(args.out_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("Family-Aware Link Prediction Evaluation\n")
            f.write("=" * 50 + "\n\n")
            f.write("Holds out ALL edges for test MOF families.\n")
            f.write(f"Seeds evaluated per method: {seeds}\n\n")
            for row in summary_rows:
                f.write(
                    f"{row['method']} (n={row['n_seeds']}): "
                    f"AUC={row['AUC_mean']:.4f}±{row['AUC_std']:.4f}, "
                    f"Hits@10={row['Hits@10_mean']:.4f}±{row['Hits@10_std']:.4f}\n"
                )
        log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
