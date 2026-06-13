"""
MOF Knowledge Graph Construction Pipeline

This script orchestrates the entire process of building the MOF Knowledge Graph:
1. Extraction: Extracts entities and relationships from raw data sources.
2. Normalization: Cleans, deduplicates, and standardizes data.
3. Construction: Builds the RDF Knowledge Graph (Turtle format).
4. Enrichment: Applies reasoning and inference to enrich the graph.

Usage:
    python build_kg.py
"""

import sys
import json
from pathlib import Path
import time

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from kg_construction.extractors.chemunity_extractor import ChemUnityExtractor
from kg_construction.extractors.digimof_extractor import DigiMOFExtractor
from kg_construction.extractors.materialsproject_extractor import MaterialsProjectExtractor
from kg_construction.extractors.opendac25_extractor import OpenDAC25Extractor
from kg_construction.extractors.linker_extractor import LinkerExtractor
from kg_construction.extractors.stability_extractor import StabilityExtractor
from kg_construction.extractors.synmof_extractor import SynMOFExtractor

from kg_construction.processing.normalizer import Normalizer
from kg_construction.construction.kg_builder import KGBuilder
from kg_construction.enrichment.enrich_kg import enrich_knowledge_graph


def _validate_required_paths(path_map):
    """Fail fast with actionable messages when required inputs are missing."""
    missing = []
    for label, p in path_map.items():
        if not Path(p).exists():
            missing.append((label, p))
    if missing:
        print("\nERROR: Required input paths are missing:")
        for label, p in missing:
            print(f"  - {label}: {p}")
        raise FileNotFoundError(
            "Missing required input data. Ensure datasets are present under data/raw."
        )


def _validate_normalized_synthesis_links(normalized_data_dir: Path, fail_on_unresolved_synmof: bool = True):
    """Validate normalized synthesis->MOF linkage and fail only for unresolved SynMOF links."""
    mofs_file = normalized_data_dir / "normalized_mofs.json"
    syn_file = normalized_data_dir / "normalized_synthesis_processes.json"
    if not mofs_file.exists() or not syn_file.exists():
        print("Warning: Skipping synthesis link validation; normalized files not found yet.")
        return

    with open(mofs_file) as f:
        mofs = json.load(f)
    with open(syn_file) as f:
        syntheses = json.load(f)

    mof_ids = {m.get("mof_id") for m in mofs if m.get("mof_id")}
    unresolved = []
    for s in syntheses:
        mof_id = s.get("mof_id")
        if mof_id and mof_id not in mof_ids:
            unresolved.append({
                "mof_id": mof_id,
                "source": s.get("data_source", "unknown"),
                "synthesis_id": s.get("synthesis_id", ""),
            })

    if not unresolved:
        print("Validation passed: all synthesis processes link to known MOF IDs.")
        return

    by_source = {}
    for rec in unresolved:
        by_source.setdefault(rec["source"], []).append(rec)

    print(f"Validation warning: {len(unresolved)} unresolved synthesis MOF IDs found.")
    for source, recs in sorted(by_source.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"  - {source}: {len(recs)} unresolved")
        for rec in recs[:10]:
            print(f"      {rec['synthesis_id']} -> {rec['mof_id']}")

    unresolved_synmof = by_source.get("SynMOF", [])
    if unresolved_synmof and fail_on_unresolved_synmof:
        raise ValueError(
            f"Unresolved SynMOF synthesis->MOF links detected ({len(unresolved_synmof)} records). "
            "Fix SynMOF ID normalization/mapping before KG construction."
        )


def _filter_synmof_to_known_mofs(sm_syn, sm_conds, sm_solvent_rels, sm_additive_rels, known_mof_ids):
    """Keep only SynMOF records whose MOF IDs exist in known MOF entities."""
    sm_syn_valid = [s for s in sm_syn if s.mof_id in known_mof_ids]
    valid_syn_ids = {s.synthesis_id for s in sm_syn_valid}
    sm_conds_valid = [c for c in sm_conds if c.synthesis_id in valid_syn_ids]
    sm_solvent_rels_valid = [r for r in sm_solvent_rels if r.get("synthesis_id") in valid_syn_ids]
    sm_additive_rels_valid = [r for r in sm_additive_rels if r.get("synthesis_id") in valid_syn_ids]

    dropped = len(sm_syn) - len(sm_syn_valid)
    if dropped > 0:
        print(
            f"  SynMOF filter: kept {len(sm_syn_valid)} synthesis records, "
            f"dropped {dropped} unresolved MOF-linked records"
        )
    return sm_syn_valid, sm_conds_valid, sm_solvent_rels_valid, sm_additive_rels_valid


def main():
    print("=" * 80)
    print("MOF KNOWLEDGE GRAPH PIPELINE")
    print("=" * 80)
    start_time = time.time()
    
    # Paths
    # project_root is already defined above
    raw_data_dir = project_root / "data" / "raw"
    normalized_data_dir = project_root / "data" / "normalized"
    kg_output_dir = project_root / "data" / "kg"
    ontology_file = project_root / "data" / "ontology" / "MOF_EMMO_ontology.ttl"
    synmof_data_dir = raw_data_dir / "SynMOF"
    mof_free_energy_data_dir = raw_data_dir / "MOF-FreeEnergy"

    _validate_required_paths({
        "ChemUnity directory": raw_data_dir / "ChemUnity",
        "DigiMOF directory": raw_data_dir / "DigiMOF",
        "MaterialsProject file": raw_data_dir / "MaterialsProjQMOF" / "MaterialsProject_cleaned.json",
        "OpenDAC25 file": raw_data_dir / "OpenDAC25" / "mof_analysis_final.json",
        "SynMOF_A.csv": synmof_data_dir / "SynMOF_A.csv",
        "Synmof_M_210618.csv": synmof_data_dir / "Synmof_M_210618.csv",
        "Synmof_Me_210618.csv": synmof_data_dir / "Synmof_Me_210618.csv",
        "MOF-FreeEnergy fe_atom dir": mof_free_energy_data_dir / "fe_atom",
        "MOF-FreeEnergy se_atom dir": mof_free_energy_data_dir / "se_atom",
        "Ontology file": ontology_file,
    })
    
    # =========================================================================
    # Phase 1: Extraction
    # =========================================================================
    print("\n[Phase 1] Extracting data from sources...")
    
    # Initialize extractors
    chemunity = ChemUnityExtractor(raw_data_dir / "ChemUnity")
    digimof = DigiMOFExtractor(raw_data_dir / "DigiMOF")
    mp = MaterialsProjectExtractor(raw_data_dir / "MaterialsProjQMOF" / "MaterialsProject_cleaned.json")
    opendac = OpenDAC25Extractor(raw_data_dir / "OpenDAC25" / "mof_analysis_final.json")
    linker_extractor = LinkerExtractor(raw_data_dir)
    
    # --- ChemUnity ---
    print("\n  Running ChemUnity Extractor...")
    cu_mofs = chemunity.extract_mofs()
    cu_props = chemunity.extract_properties()
    cu_exp_props = chemunity.extract_experimental_properties()
    cu_sgs, cu_sgs_rels = chemunity.extract_space_groups()
    cu_css, cu_css_rels = chemunity.extract_crystal_systems()
    cu_lps = chemunity.extract_lattice_parameters()
    cu_tops = chemunity.extract_topologies()
    cu_caps = chemunity.extract_capabilities()
    cu_clusters, cu_clusters_rels = chemunity.extract_metal_clusters()
    
    # --- DigiMOF ---
    print("\n  Running DigiMOF Extractor...")
    dm_mofs = digimof.extract_mofs()
    dm_clusters, dm_clusters_rels = digimof.extract_metal_clusters()
    dm_props = digimof.extract_properties()
    dm_syn = digimof.extract_synthesis()
    dm_conds = digimof.extract_synthesis_conditions()
    dm_abs = digimof.extract_abstracts()
    dm_tops, dm_tops_rels = digimof.extract_topologies()
    
    # --- MaterialsProject ---
    print("\n  Running MaterialsProject Extractor...")
    mp_mofs = mp.extract_mofs()
    mp_props = mp.extract_properties()
    mp_clusters, mp_clusters_rels = mp.extract_metal_clusters()
    mp_tops = mp.extract_topologies()
    mp_sgs, mp_sgs_rels = mp.extract_space_groups()
    mp_css, mp_css_rels = mp.extract_crystal_systems()
    
    # --- OpenDAC25 ---
    print("\n  Running OpenDAC25 Extractor...")
    od_parent_mofs = opendac.extract_parent_mofs()
    od_func_mofs = opendac.extract_functionalized_mofs()
    od_funcs = opendac.extract_functionalizations()
    od_props = opendac.extract_properties()
    od_chems = opendac.extract_functional_groups()
    
    # --- Linkers ---
    print("\n  Running Linker Extractor (Consolidated)...")
    all_linkers, linker_rels = linker_extractor.extract_all_linkers()
    
    # --- Stability Data ---
    print("\n  Running Stability Extractor...")
    # Aggregate all MOFs to find matches
    all_mofs = cu_mofs + dm_mofs + mp_mofs + od_parent_mofs
    stability_extractor = StabilityExtractor(mof_free_energy_data_dir)
    # Pass linkers for structural matching
    new_stability_mofs, stability_props = stability_extractor.extract_properties(all_mofs, all_linkers, linker_rels)

    # --- SynMOF Literature Synthesis Data ---
    print("\n  Running SynMOF Extractor...")
    synmof_extractor = SynMOFExtractor(synmof_data_dir)
    sm_syn = synmof_extractor.extract_synthesis_processes()
    sm_conds = synmof_extractor.extract_synthesis_conditions()
    sm_solvents, sm_solvent_rels = synmof_extractor.extract_solvents()
    sm_additives, sm_additive_rels = synmof_extractor.extract_additives()
    known_mof_ids = {m.mof_id for m in all_mofs + new_stability_mofs}
    sm_syn, sm_conds, sm_solvent_rels, sm_additive_rels = _filter_synmof_to_known_mofs(
        sm_syn, sm_conds, sm_solvent_rels, sm_additive_rels, known_mof_ids
    )

    # =========================================================================
    # Phase 2: Normalization
    # =========================================================================
    print("\n[Phase 2] Normalizing entities...")
    
    normalizer = Normalizer(input_dir=None, output_dir=normalized_data_dir)
    
    # Load MOFs
    normalizer.load_entities(all_mofs + new_stability_mofs, 'MOF')
    
    # Load Linkers
    normalizer.load_entities(all_linkers, 'Linker')
    
    # Load Properties
    normalizer.load_entities(cu_props + cu_exp_props + dm_props + mp_props + od_props + stability_props, 'Property')
    
    # Load Other Entities
    normalizer.load_entities(cu_clusters + dm_clusters + mp_clusters, 'MetalCluster')
    normalizer.load_entities(cu_tops + dm_tops + mp_tops, 'Topology')
    normalizer.load_entities(cu_sgs + mp_sgs, 'SpaceGroup')
    normalizer.load_entities(cu_css + mp_css, 'CrystalSystem')
    normalizer.load_entities(cu_lps, 'LatticeParameter')
    normalizer.load_entities(dm_syn + sm_syn, 'SynthesisProcess')
    normalizer.load_entities(dm_conds + sm_conds, 'SynthesisCondition')
    normalizer.load_entities(dm_abs, 'Abstract')
    normalizer.load_entities(cu_caps, 'Capability')
    normalizer.load_entities(od_func_mofs, 'FunctionalizedMOF')
    normalizer.load_entities(od_funcs, 'Functionalization')
    normalizer.load_entities(od_chems, 'Chemical')
    normalizer.load_entities(sm_solvents, 'Solvent')
    normalizer.load_entities(sm_additives, 'Additive')
    
    # Load Relationships
    print(f"  Loading {len(linker_rels)} explicit linker relationships...")
    normalizer.load_relationships(linker_rels, 'has_linker')
    
    # Load Metal Node Relationships
    metal_node_rels = cu_clusters_rels + dm_clusters_rels + mp_clusters_rels
    print(f"  Loading {len(metal_node_rels)} explicit metal node relationships...")
    normalizer.load_relationships(metal_node_rels, 'has_metal_node')
    
    # Load Space Group Relationships
    space_group_rels = cu_sgs_rels + mp_sgs_rels
    print(f"  Loading {len(space_group_rels)} explicit space group relationships...")
    normalizer.load_relationships(space_group_rels, 'has_space_group')
    
    # Load Crystal System Relationships
    crystal_system_rels = cu_css_rels + mp_css_rels
    print(f"  Loading {len(crystal_system_rels)} explicit crystal system relationships...")
    normalizer.load_relationships(crystal_system_rels, 'has_crystal_system')
    
    # Load Topology Relationships (DigiMOF)
    # ChemUnity and MP topologies are handled via MOFEntity.topology field in Normalizer
    # But DigiMOF extractor returns explicit ones now too
    print(f"  Loading {len(dm_tops_rels)} explicit topology relationships from DigiMOF...")
    normalizer.load_relationships(dm_tops_rels, 'has_topology')
    
    # Load Solvent Relationships (SynMOF)
    print(f"  Loading {len(sm_solvent_rels)} explicit solvent relationships from SynMOF...")
    normalizer.load_relationships(sm_solvent_rels, 'uses_solvent')
    
    # Load Additive Relationships (SynMOF)
    print(f"  Loading {len(sm_additive_rels)} explicit additive relationships from SynMOF...")
    normalizer.load_relationships(sm_additive_rels, 'uses_additive')
    
    # Resolve and Save
    normalizer.resolve_relationships()
    normalizer.save_normalized()
    _validate_normalized_synthesis_links(normalized_data_dir, fail_on_unresolved_synmof=True)
    
    # =========================================================================
    # Phase 3: Construction
    # =========================================================================
    print("\n[Phase 3] Building Knowledge Graph...")
    
    builder = KGBuilder(normalized_data_dir, ontology_file)
    builder.load_ontology()
    builder.load_and_add_entities()
    builder.add_relationships_from_entities()
    builder.load_and_add_predicates()
    builder.save_graph(kg_output_dir)
    
    # =========================================================================
    # Phase 4: Enrichment
    # =========================================================================
    print("\n[Phase 4] Enriching Knowledge Graph...")
    
    enrich_result = enrich_knowledge_graph(
        base_kg_path=kg_output_dir / "mof_kg.ttl",
        ontology_path=ontology_file,
        output_dir=kg_output_dir / "enriched",
        interactive=False
    )
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE in {elapsed:.1f} seconds")
    print("=" * 80)
    print(f"Enriched KG saved to: {enrich_result['output_dir']}")


if __name__ == "__main__":
    main()
