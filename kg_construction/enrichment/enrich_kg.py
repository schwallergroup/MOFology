"""
Main Ontology Enrichment Pipeline

Orchestrates the enrichment process: OWL-RL reasoning, value-based inference,
verification, and output generation.
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime
from rdflib import Graph, RDF, URIRef, Literal

import sys
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from enrichment.reasoner import apply_owlrl_reasoning
from enrichment.value_inference import apply_value_based_inference
from enrichment.enrichment_stats import EnrichmentStats
from enrichment.verification import verify_enrichment
from construction.namespace_manager import setup_namespaces, MOF_NS


def enrich_knowledge_graph(
    base_kg_path: Path = None,
    ontology_path: Path = None,
    output_dir: Path = None,
    interactive: bool = False  # Default to False for pipeline
) -> Dict:
    """
    Main enrichment pipeline.
    
    Args:
        base_kg_path: Path to base KG TTL file
        ontology_path: Path to ontology TTL file
        output_dir: Output directory
        interactive: If True, prompt user for confirmation before saving
        
    Returns:
        Dictionary with enrichment results and statistics
    """
    # Set default paths if not provided
    project_root = Path(__file__).parent.parent.parent
    if base_kg_path is None:
        base_kg_path = project_root / "data" / "kg" / "mof_kg.ttl"
    
    if ontology_path is None:
        ontology_path = project_root.parent / "Ontology" / "MOF_EMMO_master.ttl"
    
    if output_dir is None:
        output_dir = project_root / "data" / "kg" / "enriched"
    
    print("=" * 80)
    print("MOF KNOWLEDGE GRAPH ONTOLOGY ENRICHMENT")
    print("=" * 80)
    print(f"\nBase KG: {base_kg_path}")
    print(f"Ontology: {ontology_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Verify input files exist
    if not base_kg_path.exists():
        raise FileNotFoundError(f"Base KG not found: {base_kg_path}")
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology not found: {ontology_path}")
    
    # Initialize statistics tracker
    stats_tracker = EnrichmentStats()
    
    # Load base graph for statistics
    print("Loading base KG for statistics...")
    base_graph = Graph()
    base_graph.parse(str(base_kg_path), format='turtle')
    stats_tracker.collect_base_stats(base_graph)
    
    # Step 1: Apply OWL-RL reasoning
    enriched_graph, owlrl_stats = apply_owlrl_reasoning(base_kg_path, ontology_path)
    
    # Step 2: Apply value-based inference
    enriched_graph, value_stats = apply_value_based_inference(enriched_graph)
    
    # Step 3: Collect enriched statistics
    stats_tracker.collect_enriched_stats(enriched_graph, owlrl_stats, value_stats)
    stats_tracker.print_summary()
    
    # Step 4: Verification and sanity checks
    verification_results = verify_enrichment(base_graph, enriched_graph, stats_tracker.get_summary())
    
    # Step 5: Display results and get user confirmation (if interactive)
    if interactive:
        print("\n" + "=" * 80)
        print("ENRICHMENT COMPLETE - REVIEW RESULTS")
        print("=" * 80)
        # ... (omitted display code for brevity, logic remains same) ...
        # Prompt for confirmation
        print("\n" + "-" * 80)
        response = input("Save enriched KG? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y']:
            print("Enrichment cancelled by user.")
            return {
                'status': 'cancelled',
                'stats': stats_tracker.get_summary(),
                'verification': verification_results,
            }
    
    # Step 6: Save enriched KG
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SAVING ENRICHED KNOWLEDGE GRAPH")
    print("=" * 80)
    
    # Save as Turtle
    ttl_file = output_dir / "mof_kg_enriched.ttl"
    try:
        enriched_graph.serialize(destination=str(ttl_file), format='turtle')
        print(f"\n✓ Saved enriched KG as Turtle: {ttl_file}")
        print(f"  Size: {ttl_file.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"\n❌ Error saving TTL: {e}")
        raise
    
    # Save as human-readable JSON
    json_file = output_dir / "mof_kg_enriched.json"
    try:
        _save_human_readable_json(enriched_graph, json_file)
        print(f"✓ Saved enriched KG as JSON: {json_file}")
        print(f"  Size: {json_file.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        # Don't raise - JSON is optional
    
    # Save enrichment statistics
    stats_file = output_dir / "enrichment_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_tracker.get_summary(), f, indent=2, ensure_ascii=False)
    print(f"✓ Saved enrichment statistics: {stats_file}")
    
    # Save verification report
    verification_file = output_dir / "verification_report.json"
    verification_report = {
        'timestamp': datetime.now().isoformat(),
        'base_kg_stats': stats_tracker.base_stats,
        'enriched_kg_stats': stats_tracker.enriched_stats,
        'sanity_checks': {
            'data_integrity': verification_results['data_integrity']['status'],
            'inference_validation': verification_results['inference_validation']['status'],
            'statistical_checks': verification_results['statistical_checks']['status'],
            'consistency_checks': verification_results['consistency_checks']['status'],
            'output_verification': verification_results['output_verification']['status'],
        },
        'sample_inferences': verification_results.get('sample_inferences', {}),
        'warnings': verification_results.get('warnings', []),
        'errors': verification_results.get('errors', []),
        'overall_status': verification_results['overall_status'],
    }
    
    with open(verification_file, 'w') as f:
        json.dump(verification_report, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved verification report: {verification_file}")
    
    print("\n" + "=" * 80)
    print("ONTOLOGY ENRICHMENT COMPLETE")
    print("=" * 80)
    
    return {
        'status': 'success',
        'output_dir': str(output_dir),
        'stats': stats_tracker.get_summary(),
        'verification': verification_results,
    }


def _get_node_name(graph: Graph, uri: URIRef) -> str:
    """Get human-readable name for a node URI."""
    name_properties = [
        MOF_NS.hasCanonicalName,
        MOF_NS.hasChemicalName,
        MOF_NS.propertyName,
        MOF_NS.publicationTitle,
        MOF_NS.topologyCode,
    ]
    
    for prop in name_properties:
        names = list(graph.objects(uri, prop))
        if names:
            return str(names[0])
    
    # Fallback: extract from URI
    uri_str = str(uri)
    if '#' in uri_str:
        return uri_str.split('#')[-1]
    elif '/' in uri_str:
        return uri_str.split('/')[-1]
    return uri_str


def _get_predicate_name(predicate: URIRef) -> str:
    """Get human-readable name for a predicate URI."""
    import re
    uri_str = str(predicate)
    if '#' in uri_str:
        name = uri_str.split('#')[-1]
    elif '/' in uri_str:
        name = uri_str.split('/')[-1]
    else:
        return uri_str
    
    # Convert camelCase to readable format
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    return name


def _save_human_readable_json(graph: Graph, json_file: Path):
    """Save graph as human-readable JSON with actual names instead of URIs."""
    from rdflib import RDF, URIRef, Literal
    
    # Build name mappings for all nodes
    node_names = {}
    for s in graph.subjects():
        if isinstance(s, URIRef):
            node_names[s] = _get_node_name(graph, s)
    
    # Build predicate name mappings
    predicate_names = {}
    for p in graph.predicates():
        if isinstance(p, URIRef) and p != RDF.type:
            predicate_names[p] = _get_predicate_name(p)
    
    # Build edges list with human-readable names
    edges = []
    for s, p, o in graph:
        # Skip rdf:type triples (we'll handle types separately)
        if p == RDF.type:
            continue
        
        edge = {
            'subject': node_names.get(s, str(s)),
            'predicate': predicate_names.get(p, str(p)),
            'object': None
        }
        
        if isinstance(o, URIRef):
            edge['object'] = node_names.get(o, str(o))
        elif isinstance(o, Literal):
            edge['object'] = {
                'value': str(o),
                'datatype': str(o.datatype) if o.datatype else None,
                'language': o.language if o.language else None
            }
        else:
            edge['object'] = str(o)
        
        edges.append(edge)
    
    # Build nodes list with types
    nodes = {}
    for s, p, o in graph.triples((None, RDF.type, None)):
        if isinstance(s, URIRef):
            node_name = node_names.get(s, str(s))
            if node_name not in nodes:
                nodes[node_name] = {
                    'name': node_name,
                    'uri': str(s),
                    'types': []
                }
            type_name = str(o).split('#')[-1] if '#' in str(o) else str(o).split('/')[-1]
            if type_name not in nodes[node_name]['types']:
                nodes[node_name]['types'].append(type_name)
    
    # Create final JSON structure
    json_data = {
        'nodes': list(nodes.values()),
        'edges': edges,
        'statistics': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'total_triples': len(graph)
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point for command-line usage."""
    enrich_knowledge_graph(interactive=True)


if __name__ == "__main__":
    main()
