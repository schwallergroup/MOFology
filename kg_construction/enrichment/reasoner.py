"""
Manual Targeted Inference Engine

Applies targeted inference rules manually (property chains and capability rules)
without using full OWL-RL reasoning, which is too slow for large graphs.
"""

from pathlib import Path
from typing import Tuple, Dict
from rdflib import Graph, RDF, Literal, URIRef
from collections import defaultdict
import time

import sys
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from construction.namespace_manager import setup_namespaces, MOF_NS, SYN_NS


def apply_owlrl_reasoning(base_kg_path: Path, ontology_path: Path) -> Tuple[Graph, Dict]:
    """
    Apply manual targeted inference to the knowledge graph.
    This replaces full OWL-RL reasoning with fast, targeted rule application.
    
    Args:
        base_kg_path: Path to the base KG TTL file
        ontology_path: Path to the ontology TTL file (loaded for reference)
        
    Returns:
        Tuple of (enriched_graph, statistics_dict)
    """
    print("=" * 80)
    print("APPLYING MANUAL TARGETED INFERENCE")
    print("=" * 80)
    
    # Load base KG
    print(f"\n[1] Loading base KG from {base_kg_path}...")
    base_graph = Graph()
    base_graph.parse(str(base_kg_path), format='turtle')
    base_triples = len(base_graph)
    print(f"    Loaded {base_triples:,} triples")
    
    # Load ontology (for reference, but we won't merge it)
    print(f"\n[2] Loading ontology from {ontology_path}...")
    ontology_graph = Graph()
    ontology_graph.parse(str(ontology_path), format='turtle')
    ontology_triples = len(ontology_graph)
    print(f"    Loaded {ontology_triples:,} ontology triples (for reference)")
    
    # Setup namespaces
    setup_namespaces(base_graph)
    
    merged_triples = base_triples  # We're not merging ontology into the graph
    
    # Apply manual targeted inference
    print("\n[3] Applying manual targeted inference...")
    print(f"    Graph size: {merged_triples:,} triples")
    print("    Applying property chains and capability rules...")
    
    start_time = time.time()
    
    # Track statistics
    stats = {
        'merged_triples': merged_triples,
        'property_chain_inferences': 0,
        'capability_inferences': 0,
        'capabilities_by_type': defaultdict(int),
    }
    
    # 1. Apply Property Chains
    print("\n    [3.1] Applying property chains...")
    chain_stats = _apply_property_chains(base_graph)
    stats['property_chain_inferences'] = chain_stats['total']
    print(f"        ✓ Added {chain_stats['total']:,} property chain inferences")
    print(f"          - directlyUsesSolvent: {chain_stats['solvent']:,}")
    print(f"          - hasSynthesisCondition: {chain_stats['condition']:,}")
    print(f"          - hasSynthesisProcedure: {chain_stats['procedure']:,}")
    
    # 2. Apply Capability Inference Rules
    print("\n    [3.2] Applying capability inference rules...")
    capability_stats = _apply_capability_rules(base_graph)
    stats['capability_inferences'] = capability_stats['total']
    stats['capabilities_by_type'] = capability_stats['by_type']
    print(f"        ✓ Added {capability_stats['total']:,} capability inferences")
    for cap_type, count in sorted(capability_stats['by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"          - {cap_type}: {count:,}")
    
    elapsed_time = time.time() - start_time
    enriched_triples = len(base_graph)
    inferred_triples = enriched_triples - merged_triples
    
    stats['enriched_triples'] = enriched_triples
    stats['inferred_triples'] = inferred_triples
    stats['inverse_property_inferences'] = 0  # Skipped as requested
    stats['transitive_property_inferences'] = 0  # Skipped to avoid explosions
    stats['disjoint_class_violations'] = []
    
    print(f"\n    ✓ Inference complete in {elapsed_time:.1f} seconds")
    print(f"\n    Summary:")
    print(f"    Base triples: {base_triples:,}")
    print(f"    Enriched triples: {enriched_triples:,}")
    print(f"    Inferred triples: {inferred_triples:,}")
    
    return base_graph, stats


def _apply_property_chains(graph: Graph) -> Dict:
    """Apply property chain axioms manually."""
    stats = {
        'total': 0,
        'solvent': 0,
        'condition': 0,
        'procedure': 0,
    }
    
    # Property Chain 1: directlyUsesSolvent
    # MOF -> hasSynthesisProcess -> usesSolvent => MOF -> directlyUsesSolvent
    for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
        for syn_uri in graph.objects(mof_uri, SYN_NS.hasSynthesisProcess):
            for solvent_uri in graph.objects(syn_uri, SYN_NS.usesSolvent):
                if not any(graph.triples((mof_uri, MOF_NS.directlyUsesSolvent, solvent_uri))):
                    graph.add((mof_uri, MOF_NS.directlyUsesSolvent, solvent_uri))
                    stats['solvent'] += 1
                    stats['total'] += 1
    
    # Property Chain 2: hasSynthesisCondition
    # MOF -> hasSynthesisProcess -> hasCondition => MOF -> hasSynthesisCondition
    for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
        for syn_uri in graph.objects(mof_uri, SYN_NS.hasSynthesisProcess):
            for condition_uri in graph.objects(syn_uri, SYN_NS.hasCondition):
                if not any(graph.triples((mof_uri, MOF_NS.hasSynthesisCondition, condition_uri))):
                    graph.add((mof_uri, MOF_NS.hasSynthesisCondition, condition_uri))
                    stats['condition'] += 1
                    stats['total'] += 1
    
    # Property Chain 3: hasSynthesisProcedure
    # MOF -> hasSynthesisProcess -> hasProcedure => MOF -> hasSynthesisProcedure
    for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
        for syn_uri in graph.objects(mof_uri, SYN_NS.hasSynthesisProcess):
            for procedure_uri in graph.objects(syn_uri, SYN_NS.hasProcedure):
                if not any(graph.triples((mof_uri, MOF_NS.hasSynthesisProcedure, procedure_uri))):
                    graph.add((mof_uri, MOF_NS.hasSynthesisProcedure, procedure_uri))
                    stats['procedure'] += 1
                    stats['total'] += 1
    
    return stats


def _apply_capability_rules(graph: Graph) -> Dict:
    """Apply capability inference rules from the ontology manually."""
    stats = {
        'total': 0,
        'by_type': defaultdict(int),
    }
    
    # Track existing capabilities to avoid duplicates
    existing_capabilities = {}
    for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
        existing_capabilities[mof_uri] = set(graph.objects(mof_uri, MOF_NS.hasCapability))
    
    # Rule 1: CO2 Capture from CO2 uptake property
    print("        Checking Rule 1: CO2 Capture from CO2 uptake...")
    count = _infer_capability_from_property(
        graph, MOF_NS.hasComputationalProperty, 
        "CO2 uptake", MOF_NS.CO2CaptureCapability, existing_capabilities
    )
    stats['total'] += count
    stats['by_type']['CO2CaptureCapability'] += count
    
    # Rule 2: CO2 Capture from CO2 binding energy property
    print("        Checking Rule 2: CO2 Capture from CO2 binding energy...")
    count = _infer_capability_from_property(
        graph, MOF_NS.hasComputationalProperty,
        "CO2 binding energy", MOF_NS.CO2CaptureCapability, existing_capabilities
    )
    stats['total'] += count
    stats['by_type']['CO2CaptureCapability'] += count
    
    # Rule 3: CO2 Capture from amine functionalization
    print("        Checking Rule 3: CO2 Capture from amine functionalization...")
    count = _infer_capability_from_functionalization(
        graph, MOF_NS.CO2CaptureCapability, existing_capabilities
    )
    stats['total'] += count
    stats['by_type']['CO2CaptureCapability'] += count
    
    # Rule 4: DAC from CO2 uptake at LP
    print("        Checking Rule 4: DAC from CO2 uptake at LP...")
    count = _infer_capability_from_property(
        graph, MOF_NS.hasComputationalProperty,
        "CO2 uptake at LP", MOF_NS.DACCapability, existing_capabilities
    )
    stats['total'] += count
    stats['by_type']['DACCapability'] += count
    
    # Rule 5: CH4 Storage from CH4 properties
    print("        Checking Rule 5: CH4 Storage from CH4 properties...")
    ch4_properties = ["CH4 uptake", "CH4 storage", "CH4 delivery capacity", "CH4 high pressure storage"]
    for prop_name in ch4_properties:
        count = _infer_capability_from_property(
            graph, MOF_NS.hasComputationalProperty,
            prop_name, MOF_NS.MethaneStorageCapability, existing_capabilities
        )
        stats['total'] += count
        stats['by_type']['MethaneStorageCapability'] += count
    
    # Rule 6: H2 Storage from H2 uptake
    print("        Checking Rule 6: H2 Storage from H2 uptake...")
    count = _infer_capability_from_property(
        graph, MOF_NS.hasComputationalProperty,
        "H2 uptake", MOF_NS.HydrogenStorageCapability, existing_capabilities
    )
    stats['total'] += count
    stats['by_type']['HydrogenStorageCapability'] += count
    
    # Rule 7: Photocatalytic from band gap
    print("        Checking Rule 7: Photocatalytic from band gap...")
    count = _infer_capability_from_property(
        graph, MOF_NS.hasComputationalProperty,
        "Band gap", MOF_NS.PhotocatalyticCapability, existing_capabilities
    )
    stats['total'] += count
    stats['by_type']['PhotocatalyticCapability'] += count
    
    # Rule 8: Luminescent Sensing from luminescence properties
    print("        Checking Rule 8: Luminescent Sensing from luminescence...")
    lum_properties = ["Luminescence", "Fluorescence", "Phosphorescence"]
    for prop_name in lum_properties:
        count = _infer_capability_from_property(
            graph, MOF_NS.hasPhysicalProperty,
            prop_name, MOF_NS.LuminescentSensingCapability, existing_capabilities
        )
        stats['total'] += count
        stats['by_type']['LuminescentSensingCapability'] += count
    
    # Rule 9: Catalysis from catalytic activity
    print("        Checking Rule 9: Catalysis from catalytic activity...")
    count = _infer_capability_from_property(
        graph, MOF_NS.hasPhysicalProperty,
        "Catalytic activity", MOF_NS.CatalysisCapability, existing_capabilities
    )
    stats['total'] += count
    stats['by_type']['CatalysisCapability'] += count
    
    return stats


def _infer_capability_from_property(
    graph: Graph, 
    property_predicate: URIRef,
    property_name: str,
    capability: URIRef,
    existing_capabilities: Dict
) -> int:
    """Infer capability for MOFs that have a property with the given name."""
    count = 0
    property_name_lower = property_name.lower()
    
    for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
        # Skip if already has this capability
        if capability in existing_capabilities.get(mof_uri, set()):
            continue
        
        # Check if MOF has a property with matching name
        for prop_uri in graph.objects(mof_uri, property_predicate):
            prop_name_obj = list(graph.objects(prop_uri, MOF_NS.propertyName))
            if prop_name_obj:
                prop_name = str(prop_name_obj[0]).lower()
                if property_name_lower in prop_name or prop_name in property_name_lower:
                    # Add capability
                    graph.add((mof_uri, MOF_NS.hasCapability, capability))
                    if mof_uri not in existing_capabilities:
                        existing_capabilities[mof_uri] = set()
                    existing_capabilities[mof_uri].add(capability)
                    count += 1
                    break  # Only add once per MOF
    
    return count


def _infer_capability_from_functionalization(
    graph: Graph,
    capability: URIRef,
    existing_capabilities: Dict
) -> int:
    """Infer CO2 capture capability from amine functionalization."""
    count = 0
    
    for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
        # Skip if already has this capability
        if capability in existing_capabilities.get(mof_uri, set()):
            continue
        
        # Check if MOF has amine functionalization
        for func_uri in graph.objects(mof_uri, SYN_NS.hasFunctionalization):
            func_type = list(graph.objects(func_uri, SYN_NS.hasFunctionalizationType))
            if func_type:
                func_type_str = str(func_type[0]).lower()
                if 'amine' in func_type_str:
                    # Add capability
                    graph.add((mof_uri, MOF_NS.hasCapability, capability))
                    if mof_uri not in existing_capabilities:
                        existing_capabilities[mof_uri] = set()
                    existing_capabilities[mof_uri].add(capability)
                    count += 1
                    break  # Only add once per MOF
    
    return count