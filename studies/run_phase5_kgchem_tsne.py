#!/usr/bin/env python3
"""
run_phase5_kgchem_tsne.py
=========================
Standalone Phase-5-style script for t-SNE plots using concatenated
features: [KG embedding || chemical feature vector].

For each method (CompGCN, TransE, Node2Vec):
  - load all-entity KG embeddings
  - sample entities stratified by entity type
  - concatenate MOF-level chemical features where available
  - zero-fill chemical features for non-MOF entities
  - run PCA -> t-SNE
  - save entity-type colored scatter plot using plasma palette
"""

import gc
import logging
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="plasma")

# Import CompGCN architecture
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings"))
from CompGCN import CompGCNModel  # noqa: E402


# Config defaults
CHEM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv")
COMPGCN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings")
TRANSE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings")
N2V_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec")
KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/full_comparison)

SEED = 42
TSNE_ENT_N = 20_000


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
        return "Publication"
    if frag.startswith("SYN_") or frag.startswith("FUNC_"):
        return "Synthesis"
    if "synthesis#" in uri:
        return "Synthesis"
    if "owl#" in uri or "rdf-syntax" in uri or "rdf-schema" in uri:
        return "Ontology"
    return "Other"


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
    """Build MOF-level chemical feature matrix keyed by mof_uri."""
    df = df_prop.drop_duplicates(subset=["mof_uri"]).copy()

    numeric_cols = [
        c for c in [
            "Number of atoms",
            "Unit cell volume",
            "Space group number",
            "Density",
            "Largest cavity diameter",
            "Pore limiting diameter",
            "Band gap (PBE)",
        ] if c in df.columns
    ]
    df_num = (
        df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if numeric_cols else pd.DataFrame(index=df.index)
    )

    cat_cols = [
        c for c in ["crystal_system", "space_group", "topology", "metal_cluster_elements"]
        if c in df.columns
    ]
    df_cat = (
        pd.get_dummies(
            df[cat_cols].fillna("UNK"),
            prefix=[f"cat_{c}" for c in cat_cols],
            dtype=np.float32,
        )
        if cat_cols else pd.DataFrame(index=df.index)
    )

    elements = ["C", "H", "N", "O", "S", "Zn", "Cu", "Fe", "Al", "Zr", "Co", "Ni"]
    if "chemical_formula" in df.columns:
        df_formula = pd.DataFrame(
            df["chemical_formula"]
            .fillna("")
            .apply(lambda x: _formula_features(x, elements))
            .tolist()
        )
    else:
        df_formula = pd.DataFrame(
            [{f"frac_{e}": 0.0 for e in elements}] * len(df)
        )

    if "linker_smiles" in df.columns:
        fp = np.vstack(df["linker_smiles"].fillna("").map(_smiles_fp).values)
    else:
        fp = np.zeros((len(df), 256), dtype=np.float32)
    df_fp = pd.DataFrame(fp, columns=[f"fp_linker_{i}" for i in range(fp.shape[1])])

    out = pd.concat(
        [df_num.reset_index(drop=True), df_cat.reset_index(drop=True), df_formula, df_fp],
        axis=1,
    )
    out.insert(0, "mof_uri", df["mof_uri"].values)
    out = out.set_index("mof_uri").apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out.astype(np.float32)


def load_compgcn_embeddings(device: torch.device) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load CompGCN embeddings from saved model by running one forward pass."""
    log.info("  -- CompGCN: loading saved model --")
    ent2id: Dict[str, int] = torch.load(
        os.path.join(COMPGCN_DIR, "ent2id.pt"), weights_only=False
    )
    rel2id: Dict[str, int] = torch.load(
        os.path.join(COMPGCN_DIR, "rel2id.pt"), weights_only=False
    )

    ckpt_candidates = [
        os.path.join(COMPGCN_DIR, "compgcn_final_model.pt"),
        os.path.join(COMPGCN_DIR, "compgcn_best_model.pt"),
        os.path.join(COMPGCN_DIR, "best_model.pt"),
    ]
    ckpt_candidates = [p for p in ckpt_candidates if os.path.exists(p)]
    if not ckpt_candidates:
        raise FileNotFoundError("No CompGCN checkpoint found in gnn_embeddings.")

    checkpoint = None
    chosen_ckpt = None
    for ckpt_path in ckpt_candidates:
        candidate = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in candidate:
            checkpoint = candidate
            chosen_ckpt = ckpt_path
            break
    if checkpoint is None:
        raise RuntimeError("No valid CompGCN checkpoint with model_state_dict was found.")

    # Prefer training-time args when available (matches existing full-study flow).
    args = checkpoint.get("args", {})
    num_entities = checkpoint.get("num_entities", len(ent2id))
    num_relations = checkpoint.get("num_relations", len(rel2id))
    emb_dim = int(args.get("emb_dim", checkpoint.get("emb_dim", 256)))
    num_layers = int(args.get("num_layers", checkpoint.get("num_layers", 2)))
    dropout = float(args.get("dropout", checkpoint.get("dropout", 0.1)))
    comp_op = args.get("comp_op", checkpoint.get("comp_op", "mult"))
    decoder = args.get("decoder", checkpoint.get("decoder", "distmult"))
    num_bases = int(args.get("num_bases", checkpoint.get("num_bases", 4)))
    log.info("    Checkpoint: %s", chosen_ckpt)
    log.info(
        "    Config: %d ent, %d rel, %dd, %d layers, %d bases",
        num_entities, num_relations, emb_dim, num_layers, num_bases,
    )

    log.info("    Rebuilding edge_index from KG ...")
    import rdflib
    from rdflib import Graph as RDFGraph

    rdf_graph = RDFGraph()
    rdf_graph.parse(KG_PATH, format="turtle")

    src_list: List[int] = []
    dst_list: List[int] = []
    et_list: List[int] = []
    for s, p, o in rdf_graph:
        if not isinstance(o, (rdflib.URIRef, rdflib.BNode)):
            continue
        s_id = ent2id.get(str(s))
        o_id = ent2id.get(str(o))
        p_str = str(p)
        r_id = rel2id.get(p_str)
        if s_id is None or o_id is None or r_id is None:
            continue
        src_list.append(s_id)
        dst_list.append(o_id)
        et_list.append(r_id)
        inv_r = rel2id.get(f"{p_str}__inverse")
        if inv_r is not None:
            src_list.append(o_id)
            dst_list.append(s_id)
            et_list.append(inv_r)

    del rdf_graph
    gc.collect()

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).to(device)
    edge_type = torch.tensor(et_list, dtype=torch.long).to(device)

    model = None
    load_error = None
    for ckpt_path in ckpt_candidates:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" not in ckpt:
            continue
        args_i = ckpt.get("args", {})
        model_i = CompGCNModel(
            num_entities=ckpt.get("num_entities", len(ent2id)),
            num_relations=ckpt.get("num_relations", len(rel2id)),
            emb_dim=int(args_i.get("emb_dim", ckpt.get("emb_dim", emb_dim))),
            num_layers=int(args_i.get("num_layers", ckpt.get("num_layers", num_layers))),
            dropout=float(args_i.get("dropout", ckpt.get("dropout", dropout))),
            comp_op=args_i.get("comp_op", ckpt.get("comp_op", comp_op)),
            decoder=args_i.get("decoder", ckpt.get("decoder", decoder)),
            num_bases=int(args_i.get("num_bases", ckpt.get("num_bases", num_bases))),
        ).to(device)
        try:
            model_i.load_state_dict(ckpt["model_state_dict"])
            checkpoint = ckpt
            chosen_ckpt = ckpt_path
            model = model_i
            break
        except RuntimeError as exc:
            load_error = exc
            del model_i
            torch.cuda.empty_cache()
            continue

    if model is None:
        raise RuntimeError(
            f"Failed to load any CompGCN checkpoint. Last error: {load_error}"
        )
    log.info("    Loaded model weights from: %s", chosen_ckpt)
    model.eval()

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
    """Load TransE all-entity embeddings."""
    log.info("  -- TransE: loading saved model --")
    ent2id: Dict[str, int] = torch.load(
        os.path.join(TRANSE_DIR, "ent2id.pt"), weights_only=False
    )
    checkpoint = torch.load(
        os.path.join(TRANSE_DIR, "transe_best_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    all_emb = checkpoint["model_state_dict"]["entity_emb.weight"]
    log.info("    TransE embeddings: %s", list(all_emb.shape))
    return all_emb, ent2id


def load_node2vec_embeddings() -> Tuple[torch.Tensor, Dict[str, int]]:
    """Load Node2Vec all-entity embeddings."""
    log.info("  -- Node2Vec: loading saved 256d embeddings --")
    saved = torch.load(
        os.path.join(N2V_DIR, "mof_embeddings_256d_p1.0_q1.0.pt"),
        weights_only=False,
    )
    all_emb = saved["embeddings"]
    ent2id = saved["ent2id"]
    log.info("    Node2Vec embeddings: %s", list(all_emb.shape))
    return all_emb, ent2id


def _plot_tsne(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: str,
    max_legend: int = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    unique = sorted(set(labels))

    if len(unique) > max_legend:
        counts = Counter(labels)
        top = [k for k, _ in counts.most_common(max_legend - 1)]
        labels = np.array(["Other" if l not in top else l for l in labels])
        unique = sorted(set(labels))

    palette = sns.color_palette("plasma", n_colors=max(len(unique), 1))
    for i, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[palette[i]],
            label=lab,
            alpha=0.45,
            s=4,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=4, fontsize=8, loc="best", framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("    Saved %s", out_path)


def _run_tsne(X: np.ndarray, seed: int) -> np.ndarray:
    n_pca = min(50, X.shape[1], X.shape[0] - 1)
    X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X)
    X_2d = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X_pca)
    return X_2d


def _select_stratified_indices(types: np.ndarray, n_total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = Counter(types)
    selected: List[int] = []
    total_n = len(types)
    for t, cnt in counts.items():
        frac = cnt / total_n
        n_pick = max(20, int(frac * n_total))
        candidates = np.where(types == t)[0]
        chosen = rng.choice(candidates, size=min(n_pick, len(candidates)), replace=False)
        selected.extend(chosen.tolist())
    rng.shuffle(selected)
    return np.asarray(selected[:n_total], dtype=np.int64)


def _build_concat_matrix(
    emb_tensor: torch.Tensor,
    ent2id: Dict[str, int],
    chem_df: pd.DataFrame,
    sample_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    uri_list = list(ent2id.keys())
    uri_selected = [uri_list[i] for i in sample_positions]
    id_selected = [ent2id[u] for u in uri_selected]

    kg = emb_tensor[id_selected].numpy().astype(np.float32)
    chem_dim = chem_df.shape[1]
    zero_vec = np.zeros(chem_dim, dtype=np.float32)

    chem_rows = []
    for uri in uri_selected:
        if uri in chem_df.index:
            chem_rows.append(chem_df.loc[uri].values.astype(np.float32))
        else:
            chem_rows.append(zero_vec)
    chem = np.vstack(chem_rows).astype(np.float32)
    labels = np.array([infer_entity_type(u) for u in uri_selected])
    X = np.hstack([kg, chem]).astype(np.float32)
    return X, labels


def run_entity_tsne_kgchem(
    all_entity_embs: Dict[str, Tuple[torch.Tensor, Dict[str, int]]],
    chem_df: pd.DataFrame,
    out_dir: str,
    tsne_ent_n: int,
    seed: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    log.info("=" * 70)
    log.info("KG+CHEM ENTITY t-SNE")
    log.info("=" * 70)
    log.info("Chemical feature dim: %d", chem_df.shape[1])

    for method_name, (emb_tensor, ent2id) in all_entity_embs.items():
        uri_list = list(ent2id.keys())
        types = np.array([infer_entity_type(u) for u in uri_list])
        n_total = min(tsne_ent_n, len(uri_list))
        sample_positions = _select_stratified_indices(types, n_total, seed)

        X_concat, labels = _build_concat_matrix(
            emb_tensor=emb_tensor,
            ent2id=ent2id,
            chem_df=chem_df,
            sample_positions=sample_positions,
        )
        log.info("  %s: running t-SNE on %d entities (concat dim=%d)",
                 method_name, X_concat.shape[0], X_concat.shape[1])
        X_2d = _run_tsne(X_concat, seed=seed)
        _plot_tsne(
            emb_2d=X_2d,
            labels=labels,
            title=f"{method_name}: KG+Chemical Entity Embeddings by Type",
            out_path=os.path.join(out_dir, f"tsne_entity_kgchem_{method_name.lower()}.png"),
            max_legend=10,
        )


def check_required_files() -> None:
    required = [
        CHEM_PATH,
        KG_PATH,
        os.path.join(COMPGCN_DIR, "ent2id.pt"),
        os.path.join(COMPGCN_DIR, "rel2id.pt"),
        os.path.join(TRANSE_DIR, "ent2id.pt"),
        os.path.join(TRANSE_DIR, "transe_best_model.pt"),
        os.path.join(N2V_DIR, "mof_embeddings_256d_p1.0_q1.0.pt"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        for p in missing:
            log.error("Missing file: %s", p)
        raise FileNotFoundError(f"{len(missing)} required file(s) are missing.")


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    check_required_files()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading chemical properties and building chem feature matrix ...")
    df_prop = pd.read_csv(CHEM_PATH)
    if "mof_uri" not in df_prop.columns:
        raise RuntimeError("Chemical CSV must include 'mof_uri'.")
    chem_df = build_chem_feature_df(df_prop)
    log.info("Chem features: %d MOFs x %d dims", chem_df.shape[0], chem_df.shape[1])

    log.info("Loading embeddings for all three methods ...")
    compgcn_emb, compgcn_ent2id = load_compgcn_embeddings(device)
    transe_emb, transe_ent2id = load_transe_embeddings()
    n2v_emb, n2v_ent2id = load_node2vec_embeddings()

    all_entity_embs = {
        "CompGCN": (compgcn_emb, compgcn_ent2id),
        "TransE": (transe_emb, transe_ent2id),
        "Node2Vec": (n2v_emb, n2v_ent2id),
    }

    run_entity_tsne_kgchem(
        all_entity_embs=all_entity_embs,
        chem_df=chem_df,
        out_dir=OUT_DIR,
        tsne_ent_n=TSNE_ENT_N,
        seed=SEED,
    )

    elapsed_m = (time.time() - t0) / 60
    log.info("=" * 70)
    log.info("KG+CHEM t-SNE COMPLETE  --  %.1f minutes elapsed", elapsed_m)
    log.info("Outputs written to %s", OUT_DIR)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
