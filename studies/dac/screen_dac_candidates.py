#!/usr/bin/env python3
"""DAC screening from property models, KG tasks, and embedding interpretability."""

from __future__ import annotations

import argparse
import ast
import glob
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdflib import Graph
from sklearn.metrics.pairwise import cosine_similarity

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


log = logging.getLogger(__name__)


@dataclass
class CandidatePick:
    label: str
    uri: str
    score: float
    details: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen DAC candidates from MOF KG studies outputs.")
    parser.add_argument("--kg_path", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl"))
    parser.add_argument("--chem_csv", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv"))
    parser.add_argument("--pred_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/ML_Chem/prediction_results))
    parser.add_argument("--relation_pred_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "relation"))
    parser.add_argument(
        "--relation_summary",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/relation_compare/relation_summary.csv),
    )
    parser.add_argument(
        "--relation_lp_metrics",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "link_prediction_hetero_filtered/relation_specific_metrics.csv"),
    )
    parser.add_argument(
        "--concept_metrics",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/concept_vectors/concept_probe_metrics.csv),
    )
    parser.add_argument(
        "--concept_rankings_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/concept_vectors/rankings),
    )
    parser.add_argument(
        "--compgcn_csv",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_compgcn_embeddings_256d_3layers.csv),
    )
    parser.add_argument("--config", default="studies/dac/dac_criteria.yaml")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/dac_screen))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))


def load_config(path: str) -> dict:
    default_cfg = {
        "weights": {"co2_affinity": 0.34, "hydrophobicity": 0.33, "stability": 0.33},
        "scaling": {"quantile_low": 0.05, "quantile_high": 0.95},
        "filters": {
            "min_total_confidence": 0.35,
            "top_n_real": 200,
            "top_n_hypothetical": 200,
            "top_n_cards": 3,
            "max_track_a_candidates": 20000,
        },
        "reliability": {
            "min_r2_for_trust": 0.30,
            "inferred_component_prior": 0.30,
            "relation_delta_co2": 0.85,
            "relation_delta_h2o": 0.27,
        },
    }
    if yaml is None:
        log.warning("PyYAML unavailable; using in-code default DAC criteria.")
        return default_cfg
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        return default_cfg
    # Merge shallow keys to keep script robust against partial configs.
    merged = default_cfg.copy()
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def read_relation_quality(path: str, model_name: str = "GraphGPS") -> Dict[str, float]:
    df = pd.read_csv(path)
    sub = df[df["Model"].astype(str).str.lower() == model_name.lower()]
    if sub.empty:
        sub = df
    return {str(r["Relation"]): float(r["AUC"]) for _, r in sub.iterrows()}


def read_property_reliability(pred_dir: str, min_r2_for_trust: float) -> Dict[str, float]:
    summary_path = os.path.join(pred_dir, "model_performance_summary.csv")
    df = pd.read_csv(summary_path)
    best = df.sort_values("R2", ascending=False).groupby("property", as_index=False).first()
    reliab = {}
    for _, row in best.iterrows():
        r2 = float(row["R2"])
        reliab[str(row["property"])] = r2 if r2 >= min_r2_for_trust else max(0.05, r2 * 0.2)
    return reliab


def read_relation_delta_reliability(relation_summary_path: str, cfg_rel: dict) -> Dict[str, float]:
    rel = {"Delta_CO2": float(cfg_rel.get("relation_delta_co2", 0.85)), "Delta_H2O": float(cfg_rel.get("relation_delta_h2o", 0.27))}
    try:
        df = pd.read_csv(relation_summary_path)
        df = df[(df["AmineType"] == "ALL")]
        if not df.empty:
            dco2 = df[df["Target"] == "Delta_CO2"]["R2"].max()
            dh2o = df[df["Target"] == "Delta_H2O"]["R2"].max()
            if pd.notna(dco2):
                rel["Delta_CO2"] = max(0.05, float(dco2))
            if pd.notna(dh2o):
                rel["Delta_H2O"] = max(0.05, float(dh2o))
    except Exception as exc:  # pragma: no cover
        log.warning("Could not parse relation summary (%s); using config defaults.", exc)
    return rel


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def parse_tokens(raw: object) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [tok.strip() for tok in raw.split(";") if tok.strip()]


def build_topology_frequency(df: pd.DataFrame) -> pd.Series:
    topo = df["topology"].fillna("").astype(str).str.strip() if "topology" in df.columns else pd.Series("", index=df.index)
    counts = topo[topo != ""].value_counts(normalize=True)
    out = topo.map(counts).fillna(0.0)
    return out


def robust_scale(series: pd.Series, low_q: float, high_q: float, direction: str = "higher") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(np.nan, index=s.index)
    clipped = s.clip(lower=lo, upper=hi)
    norm = (clipped - lo) / (hi - lo)
    if direction == "lower":
        norm = 1.0 - norm
    return norm


def score_real_mofs(
    df: pd.DataFrame,
    cfg: dict,
    property_reliability: Dict[str, float],
) -> pd.DataFrame:
    low_q = float(cfg["scaling"]["quantile_low"])
    high_q = float(cfg["scaling"]["quantile_high"])

    score_df = pd.DataFrame(index=df.index)
    score_df["mof_uri"] = df["mof_uri"]
    score_df["CO2 binding energy"] = _to_numeric(df, "CO2 binding energy")
    score_df["CO2 Henry constant"] = _to_numeric(df, "CO2 Henry constant")
    score_df["CO2 uptake at high pressure"] = _to_numeric(df, "CO2 uptake at high pressure")
    score_df["H2O binding energy"] = _to_numeric(df, "H2O binding energy")
    score_df["Thermal Stability"] = _to_numeric(df, "Thermal Stability")
    score_df["Density"] = _to_numeric(df, "Density")
    score_df["Pore limiting diameter"] = _to_numeric(df, "Pore limiting diameter")
    score_df["topology_frequency"] = build_topology_frequency(df)

    co2_parts = {
        "CO2 binding energy": robust_scale(score_df["CO2 binding energy"], low_q, high_q, direction="lower"),
        "CO2 Henry constant": robust_scale(np.log1p(score_df["CO2 Henry constant"]), low_q, high_q, direction="higher"),
        "CO2 uptake at high pressure": robust_scale(score_df["CO2 uptake at high pressure"], low_q, high_q, direction="higher"),
    }
    hyd_parts = {
        "H2O binding energy": robust_scale(score_df["H2O binding energy"], low_q, high_q, direction="higher"),
    }
    stab_parts = {
        "Thermal Stability": robust_scale(score_df["Thermal Stability"], low_q, high_q, direction="higher"),
        "Density": robust_scale(score_df["Density"], low_q, high_q, direction="higher"),
        "Pore limiting diameter": robust_scale(score_df["Pore limiting diameter"], low_q, high_q, direction="lower"),
        "topology_frequency": robust_scale(score_df["topology_frequency"], low_q, high_q, direction="higher"),
    }

    def weighted_avg(parts: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
        values = []
        weights = []
        for name, s in parts.items():
            w = 1.0 if name == "topology_frequency" else float(property_reliability.get(name, 1.0))
            values.append(s.fillna(0.0) * w)
            weights.append(s.notna().astype(float) * w)
        total = np.sum(values, axis=0)
        denom = np.sum(weights, axis=0)
        score = np.divide(total, denom, out=np.full_like(total, np.nan), where=denom > 0)
        conf = np.divide(denom, np.max(denom), out=np.zeros_like(denom), where=np.max(denom) > 0)
        return pd.Series(score, index=score_df.index), pd.Series(conf, index=score_df.index)

    co2_score, co2_conf = weighted_avg(co2_parts)
    hyd_score, hyd_conf = weighted_avg(hyd_parts)
    stab_score, stab_conf = weighted_avg(stab_parts)

    w_co2 = float(cfg["weights"]["co2_affinity"])
    w_hyd = float(cfg["weights"]["hydrophobicity"])
    w_stb = float(cfg["weights"]["stability"])
    total = w_co2 * co2_score + w_hyd * hyd_score + w_stb * stab_score
    conf = w_co2 * co2_conf + w_hyd * hyd_conf + w_stb * stab_conf

    out = df.copy()
    out["co2_score"] = co2_score
    out["hydrophobic_score"] = hyd_score
    out["stability_score"] = stab_score
    out["dac_score"] = total
    out["dac_confidence"] = conf
    out["criteria_pass_stable"] = out["stability_score"] >= out["stability_score"].quantile(0.60)
    out["criteria_pass_hydrophobic"] = out["hydrophobic_score"] >= out["hydrophobic_score"].quantile(0.60)
    out["criteria_pass_co2"] = out["co2_score"] >= out["co2_score"].quantile(0.60)
    return out


def build_metal_linker_pairs(df: pd.DataFrame) -> Tuple[pd.DataFrame, set]:
    rows = []
    observed = set()
    for _, r in df.iterrows():
        metals = parse_tokens(r.get("metal_cluster_elements"))
        linkers = parse_tokens(r.get("linker_smiles"))
        for m in metals:
            for l in linkers:
                rows.append({"mof_uri": r["mof_uri"], "metal": m, "linker": l})
                observed.add((m, l))
    return pd.DataFrame(rows), observed


def make_component_priors(real_ranked: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs, _ = build_metal_linker_pairs(real_ranked)
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = pairs.merge(
        real_ranked[["mof_uri", "co2_score", "hydrophobic_score", "stability_score", "dac_score"]],
        on="mof_uri",
        how="left",
    )
    metal_prior = merged.groupby("metal", as_index=False)[["co2_score", "hydrophobic_score", "stability_score", "dac_score"]].mean()
    linker_prior = merged.groupby("linker", as_index=False)[["co2_score", "hydrophobic_score", "stability_score", "dac_score"]].mean()
    metal_prior["metal_freq"] = merged["metal"].value_counts(normalize=True).reindex(metal_prior["metal"]).values
    linker_prior["linker_freq"] = merged["linker"].value_counts(normalize=True).reindex(linker_prior["linker"]).values
    return metal_prior, linker_prior


def rank_hypothetical_track_a(
    real_ranked: pd.DataFrame,
    relation_quality: Dict[str, float],
    cfg: dict,
) -> pd.DataFrame:
    pair_df, observed_pairs = build_metal_linker_pairs(real_ranked)
    if pair_df.empty:
        return pd.DataFrame()

    metal_prior, linker_prior = make_component_priors(real_ranked)
    if metal_prior.empty or linker_prior.empty:
        return pd.DataFrame()

    max_candidates = int(cfg["filters"]["max_track_a_candidates"])
    metals = metal_prior.sort_values("metal_freq", ascending=False).head(80)
    linkers = linker_prior.sort_values("linker_freq", ascending=False).head(120)
    rows = []
    has_metal_auc = float(relation_quality.get("hasMetalNode", 0.9))
    has_linker_auc = float(relation_quality.get("hasLinker", 0.9))
    inferred_reliability = float(cfg["reliability"]["inferred_component_prior"])

    for _, mrow in metals.iterrows():
        for _, lrow in linkers.iterrows():
            m = str(mrow["metal"])
            l = str(lrow["linker"])
            if (m, l) in observed_pairs:
                continue
            co2 = float(np.nanmean([mrow["co2_score"], lrow["co2_score"]]))
            hyd = float(np.nanmean([mrow["hydrophobic_score"], lrow["hydrophobic_score"]]))
            stb = float(np.nanmean([mrow["stability_score"], lrow["stability_score"]]))
            base_dac = float(np.nanmean([mrow["dac_score"], lrow["dac_score"]]))

            plausibility = (np.sqrt(float(mrow["metal_freq"]) * float(lrow["linker_freq"]))) * np.sqrt(has_metal_auc * has_linker_auc)
            final = base_dac * plausibility
            rows.append(
                {
                    "candidate_id": f"HYP_A__{_safe(m)}__{_safe(l)}",
                    "metal": m,
                    "linker_smiles": l,
                    "co2_score": co2,
                    "hydrophobic_score": hyd,
                    "stability_score": stb,
                    "dac_score": base_dac,
                    "plausibility_score": plausibility,
                    "final_score": final,
                    "confidence": inferred_reliability * plausibility,
                    "exists_in_db": False,
                }
            )
            if len(rows) >= max_candidates:
                break
        if len(rows) >= max_candidates:
            break

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    return out


def load_relation_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "mof_uri" not in df.columns:
        return pd.DataFrame()
    return df


def choose_relation_delta_source(
    relation_pred_dir: str,
    relation_summary: str,
) -> Tuple[pd.DataFrame, str]:
    candidates = [
        ("GraphGPS", os.path.join(relation_pred_dir, "GraphGPS_dual_predictions.csv")),
        ("Node2Vec", os.path.join(relation_pred_dir, "node2vec_dual_predictions.csv")),
    ]
    best_source = "GraphGPS"
    try:
        summ = pd.read_csv(relation_summary)
        row = summ[(summ["Target"] == "Delta_CO2") & (summ["AmineType"] == "ALL")].sort_values("R2", ascending=False).head(1)
        if not row.empty:
            emb = str(row.iloc[0]["Embedding"]).lower()
            if "node2vec" in emb:
                best_source = "Node2Vec"
            elif "compgcn" in emb:
                # CompGCN per-sample deltas are not exported in current artifacts.
                # GraphGPS has per-candidate outputs and is usually more reliable than legacy node2vec.
                best_source = "GraphGPS"
    except Exception:
        pass

    for source, path in candidates:
        if source == best_source:
            df = load_relation_predictions(path)
            if not df.empty:
                return df, source
    for source, path in candidates:
        df = load_relation_predictions(path)
        if not df.empty:
            return df, source
    return pd.DataFrame(), "none"


def infer_parent_uri_from_func(func_uri: str) -> Optional[str]:
    if "FuncMOF_" not in func_uri:
        return None
    prefix = "http://emmo.info/domain-mof/mof-ontology#"
    frag = func_uri.split("#")[-1]
    if not frag.startswith("FuncMOF_"):
        return None
    body = frag.replace("FuncMOF_", "", 1)
    parent_code = body.split("_")[0]
    return f"{prefix}MOF_{parent_code}"


def infer_amine_from_func(func_uri: str) -> str:
    frag = func_uri.split("#")[-1]
    body = frag.replace("FuncMOF_", "", 1)
    parts = body.split("_")
    return parts[-1] if len(parts) > 1 else "unknown"


def rank_hypothetical_track_b(
    real_ranked: pd.DataFrame,
    relation_pred_dir: str,
    relation_summary: str,
    delta_reliability: Dict[str, float],
) -> Tuple[pd.DataFrame, str]:
    pred_df, source = choose_relation_delta_source(relation_pred_dir, relation_summary)
    if pred_df.empty:
        return pd.DataFrame(), source

    parent_lookup = real_ranked.set_index("mof_uri")
    rows = []
    for _, row in pred_df.iterrows():
        child_uri = str(row["mof_uri"])
        parent_uri = infer_parent_uri_from_func(child_uri)
        if parent_uri is None or parent_uri not in parent_lookup.index:
            continue
        parent = parent_lookup.loc[parent_uri]

        dco2 = float(row.get("pred_delta_CO2", np.nan))
        dh2o = float(row.get("pred_delta_H2O", np.nan))
        if np.isnan(dco2) or np.isnan(dh2o):
            continue

        # More negative CO2 binding energy is better.
        # Delta target is child-parent; negative delta improves binding strength.
        co2_gain = -dco2
        # Less negative / higher H2O binding is better for hydrophobicity.
        hyd_gain = dh2o
        post_co2 = float(parent["co2_score"]) + 0.25 * co2_gain
        post_hyd = float(parent["hydrophobic_score"]) + 0.25 * hyd_gain
        post_stb = float(parent["stability_score"])
        post_dac = 0.34 * post_co2 + 0.33 * post_hyd + 0.33 * post_stb

        conf = 0.6 * float(delta_reliability["Delta_CO2"]) + 0.4 * float(delta_reliability["Delta_H2O"])
        rows.append(
            {
                "candidate_uri": child_uri,
                "parent_uri": parent_uri,
                "amine_code": infer_amine_from_func(child_uri),
                "source_model": source,
                "pred_delta_CO2": dco2,
                "pred_delta_H2O": dh2o,
                "post_co2_score": post_co2,
                "post_hydrophobic_score": post_hyd,
                "post_stability_score": post_stb,
                "post_dac_score": post_dac,
                "parent_dac_score": float(parent["dac_score"]),
                "delta_dac_score": post_dac - float(parent["dac_score"]),
                "confidence": conf,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, source
    out = out.sort_values(["post_dac_score", "delta_dac_score"], ascending=False)
    return out, source


def load_compgcn_embeddings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "mof_uri" not in df.columns:
        raise ValueError("CompGCN embedding CSV missing 'mof_uri' column.")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df[["mof_uri"] + emb_cols].set_index("mof_uri")


def nearest_analogs(uri: str, emb_df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, float]]:
    if uri not in emb_df.index:
        return []
    x = emb_df.loc[[uri]].values
    sims = cosine_similarity(x, emb_df.values).ravel()
    order = np.argsort(-sims)
    out = []
    for idx in order:
        cand_uri = emb_df.index[idx]
        if cand_uri == uri:
            continue
        out.append((cand_uri, float(sims[idx])))
        if len(out) >= top_k:
            break
    return out


def concept_evidence_for_uri(uri: str, rankings_dir: str, method: str = "CompGCN") -> List[str]:
    ev = []
    top_pattern = os.path.join(rankings_dir, f"concept_alignment_topk_{method}_*.csv")
    for path in glob.glob(top_pattern):
        try:
            sub = pd.read_csv(path)
            if "mof_uri" not in sub.columns:
                continue
            if uri in set(sub["mof_uri"].astype(str)):
                concept = os.path.basename(path).replace(f"concept_alignment_topk_{method}_", "").replace(".csv", "")
                ev.append(f"Top-aligned concept: {concept}")
        except Exception:
            continue
        if len(ev) >= 6:
            break
    return ev


def parse_kg_support(kg_path: str) -> Dict[str, Dict[str, List[str]]]:
    g = Graph()
    g.parse(kg_path, format="turtle")
    has_metal = "http://emmo.info/domain-mof/mof-ontology#hasMetalNode"
    has_linker = "http://emmo.info/domain-mof/mof-ontology#hasLinker"
    has_topo = "http://emmo.info/domain-mof/mof-ontology#hasTopology"
    support: Dict[str, Dict[str, List[str]]] = {}

    for s, p, o in g:
        p_str = str(p)
        s_str = str(s)
        o_str = str(o)
        if p_str not in {has_metal, has_linker, has_topo}:
            continue
        if s_str not in support:
            support[s_str] = {"metals": [], "linkers": [], "topologies": []}
        if p_str == has_metal:
            support[s_str]["metals"].append(o_str)
        elif p_str == has_linker:
            support[s_str]["linkers"].append(o_str)
        elif p_str == has_topo:
            support[s_str]["topologies"].append(o_str)
    return support


def write_evidence_card(path: str, title: str, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for line in lines:
            f.write(f"- {line}\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    cards_dir = os.path.join(args.out_dir, "candidate_evidence_cards")
    os.makedirs(cards_dir, exist_ok=True)

    cfg = load_config(args.config)
    relation_quality = read_relation_quality(args.relation_lp_metrics)
    property_reliability = read_property_reliability(
        args.pred_dir, float(cfg["reliability"]["min_r2_for_trust"])
    )
    delta_reliability = read_relation_delta_reliability(args.relation_summary, cfg["reliability"])
    chem_df = pd.read_csv(args.chem_csv)

    # Real MOF ranking
    real = score_real_mofs(chem_df, cfg, property_reliability)
    min_conf = float(cfg["filters"]["min_total_confidence"])
    real = real[real["dac_confidence"] >= min_conf].copy()
    real = real.sort_values("dac_score", ascending=False)
    real.to_csv(os.path.join(args.out_dir, "real_mof_ranked.csv"), index=False)
    log.info("Saved real MOF ranking (%d rows).", len(real))

    # Track A: novel metal-linker recombinations
    track_a = rank_hypothetical_track_a(real, relation_quality, cfg)
    if not track_a.empty:
        track_a.to_csv(os.path.join(args.out_dir, "hypothetical_metal_linker_ranked.csv"), index=False)
        log.info("Saved hypothetical metal-linker ranking (%d rows).", len(track_a))
    else:
        pd.DataFrame().to_csv(os.path.join(args.out_dir, "hypothetical_metal_linker_ranked.csv"), index=False)

    # Track B: functionalized derivatives
    track_b, relation_source = rank_hypothetical_track_b(
        real,
        args.relation_pred_dir,
        args.relation_summary,
        delta_reliability,
    )
    if not track_b.empty:
        track_b.to_csv(os.path.join(args.out_dir, "hypothetical_functionalized_ranked.csv"), index=False)
        log.info("Saved hypothetical functionalized ranking (%d rows) via %s deltas.", len(track_b), relation_source)
    else:
        pd.DataFrame().to_csv(os.path.join(args.out_dir, "hypothetical_functionalized_ranked.csv"), index=False)

    # Select representatives
    picks: List[CandidatePick] = []
    if not real.empty:
        rr = real.iloc[0]
        picks.append(
            CandidatePick(
                label="Real MOF",
                uri=str(rr["mof_uri"]),
                score=float(rr["dac_score"]),
                details={
                    "co2_score": float(rr["co2_score"]),
                    "hydrophobic_score": float(rr["hydrophobic_score"]),
                    "stability_score": float(rr["stability_score"]),
                    "confidence": float(rr["dac_confidence"]),
                },
            )
        )
    if not track_a.empty:
        ra = track_a.iloc[0]
        picks.append(
            CandidatePick(
                label="Hypothetical Track A (new metal+linker)",
                uri=str(ra["candidate_id"]),
                score=float(ra["final_score"]),
                details=ra.to_dict(),
            )
        )
    if not track_b.empty:
        rb = track_b.iloc[0]
        picks.append(
            CandidatePick(
                label="Hypothetical Track B (functionalized derivative)",
                uri=str(rb["candidate_uri"]),
                score=float(rb["post_dac_score"]),
                details=rb.to_dict(),
            )
        )

    compgcn_emb = load_compgcn_embeddings(args.compgcn_csv)
    kg_support = parse_kg_support(args.kg_path)

    # Summary markdown
    summary_lines = [
        "# DAC Candidate Screening Summary",
        "",
        "## Configuration",
        f"- Weights: {cfg['weights']}",
        f"- Min confidence: {cfg['filters']['min_total_confidence']}",
        f"- Relation quality (Graph task): hasMetalNode={relation_quality.get('hasMetalNode', np.nan):.3f}, hasLinker={relation_quality.get('hasLinker', np.nan):.3f}, hasFunctionalization={relation_quality.get('hasFunctionalization', np.nan):.3f}",
        f"- Relation delta reliability: {delta_reliability}",
        "",
        "## Selected Candidates",
    ]
    for pick in picks:
        summary_lines.append(f"- {pick.label}: `{pick.uri}` | score={pick.score:.4f}")

    with open(os.path.join(args.out_dir, "selected_candidates_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    # Evidence cards
    for pick in picks:
        lines = [
            f"Candidate URI/ID: `{pick.uri}`",
            f"DAC score: {pick.score:.4f}",
            f"Criteria details: {pick.details}",
        ]
        if pick.label == "Real MOF":
            uri = pick.uri
            lines.append("Type: Existing MOF in database.")
            lines.extend([f"Concept evidence: {x}" for x in concept_evidence_for_uri(uri, args.concept_rankings_dir)])
            analogs = nearest_analogs(uri, compgcn_emb, top_k=5)
            if analogs:
                lines.append(f"Nearest analogs (CompGCN cosine): {analogs}")
            if uri in kg_support:
                lines.append(
                    f"KG path evidence: hasMetalNode={len(kg_support[uri]['metals'])}, hasLinker={len(kg_support[uri]['linkers'])}, hasTopology={len(kg_support[uri]['topologies'])}"
                )
            lines.append("Risk flags: sparse direct thermal-stability labels in source table; stability includes proxy terms.")
        elif pick.label.startswith("Hypothetical Track A"):
            lines.append("Type: Novel metal+linker recombination not observed together in known MOFs.")
            lines.append("KG task usage: plausibility weighted by hasMetalNode/hasLinker link-prediction relation AUC.")
            lines.append("Risk flags: component-prior inference only; no explicit synthesized structure yet.")
        else:
            parent_uri = str(pick.details.get("parent_uri", ""))
            lines.append(f"Parent MOF: `{parent_uri}`")
            lines.append(f"Functionalization code: `{pick.details.get('amine_code', 'unknown')}`")
            lines.append(f"Relation model source: `{pick.details.get('source_model', 'unknown')}`")
            lines.append(
                "KG path evidence: candidate follows derivedFrom/hasFunctionalization pattern from existing FuncMOF nodes."
            )
            lines.append("Risk flags: Delta_H2O reliability is weaker than Delta_CO2; hydrophobic gain has higher uncertainty.")
        write_evidence_card(
            os.path.join(cards_dir, f"{_safe(pick.label)}.md"),
            pick.label,
            lines,
        )

    # Export reliability traces for transparency.
    pd.DataFrame(
        [{"property": k, "reliability_weight": v} for k, v in sorted(property_reliability.items())]
    ).to_csv(os.path.join(args.out_dir, "property_reliability_weights.csv"), index=False)
    pd.DataFrame(
        [{"relation": k, "auc": v} for k, v in sorted(relation_quality.items())]
    ).to_csv(os.path.join(args.out_dir, "kg_relation_quality.csv"), index=False)

    log.info("DAC screening complete. Outputs written to %s", args.out_dir)


if __name__ == "__main__":
    main()

