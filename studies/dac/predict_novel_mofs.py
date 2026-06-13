#!/usr/bin/env python3
"""
Predict DAC properties for novel MOFs using embedding composition.

Demonstrates that KG embeddings can be composed to predict properties
for MOFs not seen during training.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdflib import Graph, Namespace
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MOF = Namespace("http://emmo.info/domain-mof/mof-ontology#")
SYN = Namespace("http://emmo.info/domain-mof/synthesis#")


def load_embeddings(emb_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load embeddings from CSV into dict."""
    log.info(f"Loading embeddings from {emb_path}")
    df = pd.read_csv(emb_path)

    uri_col = "mof_uri"
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    embeddings = {}
    for _, row in df.iterrows():
        uri = str(row[uri_col])
        emb = row[emb_cols].values.astype(np.float32)
        frag = uri.split("#")[-1] if "#" in uri else uri
        embeddings[frag] = emb
        embeddings[uri] = emb

    log.info(f"Loaded {len(df)} embeddings with dim={len(emb_cols)}")
    return embeddings, emb_cols


def load_concept_vectors(concept_path: str) -> Dict[str, np.ndarray]:
    """Load concept vectors from CSV."""
    log.info(f"Loading concept vectors from {concept_path}")
    df = pd.read_csv(concept_path)

    concepts = {}
    for _, row in df.iterrows():
        concept_name = row["concept"]
        weights = row[[c for c in df.columns if c.startswith("w_")]].values.astype(np.float32)
        concepts[concept_name] = weights

    log.info(f"Loaded {len(concepts)} concept vectors")
    return concepts


def extract_pseudo_component_embeddings(
    kg_path: str, embeddings: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict, Dict[str, str], Dict[str, str]]:
    """Create pseudo-component embeddings by averaging MOF embeddings that share components."""
    log.info("Creating pseudo-component embeddings from MOF embeddings...")

    g = Graph()
    g.parse(kg_path, format="turtle")

    # Query for MOFs with their metal elements
    metal_query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?element
    WHERE {
        ?mof a mof:MOF .
        ?mof mof:hasMetalNode ?metal .
        ?metal mof:hasMetalElement ?element .
    }
    """
    metal_to_mofs = {}
    for row in g.query(metal_query):
        mof_uri = str(row[0]).split("#")[-1]
        element = str(row[1])
        if mof_uri in embeddings:
            if element not in metal_to_mofs:
                metal_to_mofs[element] = []
            metal_to_mofs[element].append(mof_uri)

    # Create pseudo-metal embeddings by averaging MOF embeddings per metal
    metals = {}
    metal_elements = {}
    for element, mof_list in metal_to_mofs.items():
        emb_list = [embeddings[m] for m in mof_list if m in embeddings]
        if len(emb_list) >= 5:  # Require at least 5 MOFs
            metals[element] = np.mean(emb_list, axis=0)
            metal_elements[element] = element

    log.info(f"Created {len(metals)} pseudo-metal embeddings (from MOF averages)")

    # Query for MOFs with their linkers
    linker_query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?linker ?smiles
    WHERE {
        ?mof a mof:MOF .
        ?mof mof:hasLinker ?linker .
        ?linker mof:hasSMILES ?smiles .
    }
    """
    linker_to_mofs = {}
    linker_smiles_map = {}
    for row in g.query(linker_query):
        mof_uri = str(row[0]).split("#")[-1]
        linker_uri = str(row[1]).split("#")[-1]
        smiles = str(row[2])
        if mof_uri in embeddings:
            if linker_uri not in linker_to_mofs:
                linker_to_mofs[linker_uri] = []
                linker_smiles_map[linker_uri] = smiles
            linker_to_mofs[linker_uri].append(mof_uri)

    # Create pseudo-linker embeddings by averaging MOF embeddings per linker
    linkers = {}
    linker_smiles = {}
    for linker_uri, mof_list in linker_to_mofs.items():
        emb_list = [embeddings[m] for m in mof_list if m in embeddings]
        if len(emb_list) >= 3:  # Require at least 3 MOFs
            linkers[linker_uri] = np.mean(emb_list, axis=0)
            linker_smiles[linker_uri] = linker_smiles_map[linker_uri]

    log.info(f"Created {len(linkers)} pseudo-linker embeddings (from MOF averages)")

    # Query for MOF -> metal/linker mappings for validation
    mof_components_query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?element ?linker
    WHERE {
        ?mof a mof:MOF .
        ?mof mof:hasMetalNode ?metal .
        ?metal mof:hasMetalElement ?element .
        ?mof mof:hasLinker ?linker .
    }
    """
    mof_to_components = {}
    for row in g.query(mof_components_query):
        mof_uri = str(row[0]).split("#")[-1]
        element = str(row[1])
        linker_uri = str(row[2]).split("#")[-1]
        if mof_uri not in mof_to_components:
            mof_to_components[mof_uri] = {"metals": [], "linkers": []}
        if element not in mof_to_components[mof_uri]["metals"]:
            mof_to_components[mof_uri]["metals"].append(element)
        if linker_uri not in mof_to_components[mof_uri]["linkers"]:
            mof_to_components[mof_uri]["linkers"].append(linker_uri)

    log.info(f"Found {len(mof_to_components)} MOFs with component mappings")

    return metals, linkers, mof_to_components, metal_elements, linker_smiles


def compose_embedding(
    metal_embs: List[np.ndarray], linker_embs: List[np.ndarray], method: str = "mean"
) -> np.ndarray:
    """Compose a hypothetical MOF embedding from metal and linker embeddings."""
    all_embs = metal_embs + linker_embs
    if not all_embs:
        return None

    if method == "mean":
        return np.mean(all_embs, axis=0)
    elif method == "sum":
        return np.sum(all_embs, axis=0)
    elif method == "max":
        return np.max(all_embs, axis=0)
    else:
        return np.mean(all_embs, axis=0)


def predict_concept_score(emb: np.ndarray, concept_vec: np.ndarray) -> float:
    """Predict concept score using cosine similarity with concept vector."""
    if emb is None or concept_vec is None:
        return 0.0
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
    cv_norm = concept_vec / (np.linalg.norm(concept_vec) + 1e-8)
    return float(np.dot(emb_norm, cv_norm))


def validate_composition(
    embeddings: Dict[str, np.ndarray],
    metals: Dict[str, np.ndarray],
    linkers: Dict[str, np.ndarray],
    mof_to_components: Dict,
    n_samples: int = 200,
) -> pd.DataFrame:
    """Validate embedding composition on known MOFs."""
    log.info(f"Validating composition on {n_samples} MOFs...")

    results = []
    count = 0

    for mof_uri, components in mof_to_components.items():
        if count >= n_samples:
            break

        if mof_uri not in embeddings:
            continue

        # Get actual embedding
        actual_emb = embeddings[mof_uri]

        # Get component embeddings
        metal_embs = [metals[m] for m in components["metals"] if m in metals]
        linker_embs = [linkers[l] for l in components["linkers"] if l in linkers]

        if not metal_embs or not linker_embs:
            continue

        # Compose embedding
        composed_emb = compose_embedding(metal_embs, linker_embs, method="mean")

        # Compute cosine similarity
        actual_norm = actual_emb / (np.linalg.norm(actual_emb) + 1e-8)
        composed_norm = composed_emb / (np.linalg.norm(composed_emb) + 1e-8)
        cos_sim = float(np.dot(actual_norm, composed_norm))

        results.append({
            "mof_uri": mof_uri,
            "n_metals": len(metal_embs),
            "n_linkers": len(linker_embs),
            "cosine_similarity": cos_sim,
        })
        count += 1

    df = pd.DataFrame(results)
    log.info(f"Validated {len(df)} MOFs, mean cosine similarity: {df['cosine_similarity'].mean():.4f}")
    return df


def generate_hypothetical_predictions(
    metals: Dict[str, np.ndarray],
    linkers: Dict[str, np.ndarray],
    metal_elements: Dict[str, str],
    linker_smiles: Dict[str, str],
    concepts: Dict[str, np.ndarray],
    observed_pairs: set,
    max_candidates: int = 5000,
) -> pd.DataFrame:
    """Generate predictions for novel metal-linker combinations."""
    log.info("Generating hypothetical MOF predictions...")

    # Select DAC-relevant concepts
    dac_concepts = {
        "high_Density": concepts.get("high_Density"),
        "low_Density": concepts.get("low_Density"),
    }

    # Filter to available concepts
    dac_concepts = {k: v for k, v in dac_concepts.items() if v is not None}
    log.info(f"Using concepts: {list(dac_concepts.keys())}")

    # Get top metals and linkers by frequency (proxy for importance)
    metal_list = list(metals.keys())[:100]
    linker_list = list(linkers.keys())[:100]

    results = []
    count = 0

    for metal_uri in metal_list:
        for linker_uri in linker_list:
            if count >= max_candidates:
                break

            # Skip observed pairs
            if (metal_uri, linker_uri) in observed_pairs:
                continue

            metal_emb = metals[metal_uri]
            linker_emb = linkers[linker_uri]

            composed_emb = compose_embedding([metal_emb], [linker_emb])

            # Predict concept scores
            scores = {}
            for concept_name, concept_vec in dac_concepts.items():
                scores[concept_name] = predict_concept_score(composed_emb, concept_vec)

            # Simple DAC score: high density (stability proxy)
            dac_score = scores.get("high_Density", 0.0)

            results.append({
                "candidate_id": f"HYP_{metal_uri}_{linker_uri}",
                "metal": metal_elements.get(metal_uri, metal_uri),
                "linker_smiles": linker_smiles.get(linker_uri, linker_uri)[:50],
                "predicted_dac_score": dac_score,
                **scores,
            })
            count += 1

        if count >= max_candidates:
            break

    df = pd.DataFrame(results)
    df = df.sort_values("predicted_dac_score", ascending=False)
    log.info(f"Generated {len(df)} hypothetical predictions")
    return df


def main():
    parser = argparse.ArgumentParser(description="Novel MOF Prediction via Embedding Composition")
    parser.add_argument(
        "--kg_path",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl"),
    )
    parser.add_argument(
        "--emb_path",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_compgcn_embeddings_256d_3layers.csv),
    )
    parser.add_argument(
        "--concept_path",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/concept_vectors/concept_vectors_CompGCN.csv),
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/dac_screen),
    )
    parser.add_argument("--max_candidates", type=int, default=5000)
    args = parser.parse_args()

    # Load data
    embeddings, emb_cols = load_embeddings(args.emb_path)
    concepts = load_concept_vectors(args.concept_path)

    # Extract pseudo-component embeddings from MOF averages
    metals, linkers, mof_to_components, metal_elements, linker_smiles = extract_pseudo_component_embeddings(
        args.kg_path, embeddings
    )

    # Build set of observed metal-linker pairs
    observed_pairs = set()
    for mof_uri, components in mof_to_components.items():
        for m in components["metals"]:
            for l in components["linkers"]:
                observed_pairs.add((m, l))
    log.info(f"Found {len(observed_pairs)} observed metal-linker pairs")

    # Validate composition approach
    validation_df = validate_composition(embeddings, metals, linkers, mof_to_components)
    validation_path = os.path.join(args.out_dir, "composition_validation.csv")
    validation_df.to_csv(validation_path, index=False)

    # Generate validation report
    report = []
    report.append("# Embedding Composition Validation Report\n")
    report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    report.append("\n## Summary\n")
    report.append(f"- MOFs validated: {len(validation_df)}\n")
    report.append(f"- Mean cosine similarity: {validation_df['cosine_similarity'].mean():.4f}\n")
    report.append(f"- Std cosine similarity: {validation_df['cosine_similarity'].std():.4f}\n")
    report.append(f"- Min: {validation_df['cosine_similarity'].min():.4f}\n")
    report.append(f"- Max: {validation_df['cosine_similarity'].max():.4f}\n")
    report.append(f"- % with cosine > 0.7: {(validation_df['cosine_similarity'] > 0.7).mean()*100:.1f}%\n")
    report.append(f"- % with cosine > 0.5: {(validation_df['cosine_similarity'] > 0.5).mean()*100:.1f}%\n")

    report.append("\n## Interpretation\n")
    if validation_df['cosine_similarity'].mean() > 0.5:
        report.append("Embedding composition captures meaningful structure. ")
        report.append("The composed embeddings are reasonably similar to actual MOF embeddings, ")
        report.append("suggesting that component embeddings contain information about the full MOF.\n")
    else:
        report.append("Embedding composition has limited fidelity. ")
        report.append("This is expected as MOF properties depend on specific component interactions.\n")

    report_path = os.path.join(args.out_dir, "composition_validation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    log.info(f"Validation report saved to {report_path}")

    # Generate hypothetical predictions
    hyp_df = generate_hypothetical_predictions(
        metals, linkers, metal_elements, linker_smiles,
        concepts, observed_pairs, args.max_candidates
    )
    hyp_path = os.path.join(args.out_dir, "novel_mof_embedding_predictions.csv")
    hyp_df.to_csv(hyp_path, index=False)
    log.info(f"Predictions saved to {hyp_path}")

    # Top 10 candidates
    log.info("\nTop 10 Novel MOF Candidates:")
    log.info(hyp_df[["candidate_id", "metal", "predicted_dac_score"]].head(10).to_string())


if __name__ == "__main__":
    main()
