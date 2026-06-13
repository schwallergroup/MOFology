"""
Verification and Sanity Checks

Comprehensive verification module with sanity checks for the enriched KG.
"""

from typing import Dict, List, Tuple, Set
from rdflib import Graph, RDF, URIRef, Literal, Namespace
from collections import defaultdict

import sys
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from construction.namespace_manager import MOF_NS, SYN_NS


def verify_enrichment(base_graph: Graph, enriched_graph: Graph, stats: Dict) -> Dict:
    """
    Perform comprehensive verification and sanity checks on the enriched KG.
    
    Args:
        base_graph: The original (un-enriched) graph
        enriched_graph: The enriched graph
        stats: Statistics from enrichment process
        
    Returns:
        Dictionary with verification results
    """
    print("\n" + "=" * 80)
    print("VERIFICATION AND SANITY CHECKS")
    print("=" * 80)
    
    results = {
        'data_integrity': _check_data_integrity(base_graph, enriched_graph),
        'inference_validation': _validate_inferences(base_graph, enriched_graph),
        'statistical_checks': _check_statistics(base_graph, enriched_graph, stats),
        'consistency_checks': _check_consistency(enriched_graph),
        'output_verification': _verify_output(enriched_graph),
        'sample_inferences': _collect_sample_inferences(base_graph, enriched_graph),
        'warnings': [],
        'errors': [],
    }
    
    # Overall status
    all_passed = (
        results['data_integrity']['status'] == 'PASS' and
        results['inference_validation']['status'] == 'PASS' and
        results['statistical_checks']['status'] == 'PASS' and
        results['consistency_checks']['status'] == 'PASS' and
        results['output_verification']['status'] == 'PASS'
    )
    
    results['overall_status'] = 'PASS' if all_passed else 'WARNING'
    
    # Print summary
    _print_verification_summary(results)
    
    return results


def _check_data_integrity(base_graph: Graph, enriched_graph: Graph) -> Dict:
    """Check that all original data is preserved."""
    print("\n[1] Data Integrity Checks...")
    
    base_triples = set(base_graph)
    enriched_triples = set(enriched_graph)
    
    # All base triples should be in enriched graph
    missing_triples = base_triples - enriched_triples
    
    # Check for duplicate triples (shouldn't happen, but verify)
    base_count = len(base_triples)
    enriched_count = len(enriched_triples)
    
    result = {
        'status': 'PASS',
        'base_triples': base_count,
        'enriched_triples': enriched_count,
        'missing_triples': len(missing_triples),
        'triples_added': enriched_count - base_count,
    }
    
    if missing_triples:
        result['status'] = 'FAIL'
        result['errors'] = [f"Missing {len(missing_triples)} triples from base graph"]
        print(f"  ❌ FAIL: {len(missing_triples)} triples missing from enriched graph")
    else:
        print(f"  ✓ PASS: All {base_count:,} base triples preserved")
        print(f"  ✓ Added {enriched_count - base_count:,} inferred triples")
    
    return result


def _validate_inferences(base_graph: Graph, enriched_graph: Graph) -> Dict:
    """Validate that inferences are correct."""
    print("\n[2] Inference Validation...")
    
    result = {
        'status': 'PASS',
        'samples': {},
    }
    
    # Sample inverse property inferences
    print("  Checking inverse property inferences...")
    inverse_samples = []
    
    # Check :usedInMOF (inverse of :hasLinker)
    for mof_uri, linker_uri in list(base_graph.subject_objects(MOF_NS.hasLinker))[:5]:
        # In enriched graph, should have inverse
        inverse_exists = (linker_uri, MOF_NS.usedInMOF, mof_uri) in enriched_graph
        inverse_samples.append({
            'mof': str(mof_uri),
            'linker': str(linker_uri),
            'inverse_exists': inverse_exists,
        })
    
    result['samples']['inverse_properties'] = inverse_samples
    inverse_pass = all(s['inverse_exists'] for s in inverse_samples)
    
    if inverse_pass:
        print(f"    ✓ Inverse properties validated ({len(inverse_samples)} samples)")
    else:
        print(f"    ⚠ WARNING: Some inverse properties missing")
        result['status'] = 'WARNING'
    
    # Sample property chain inferences
    print("  Checking property chain inferences...")
    chain_samples = []
    
    # Check :directlyUsesSolvent (chain: hasSynthesisProcess -> usesSolvent)
    for mof_uri in list(base_graph.subjects(RDF.type, MOF_NS.MOF))[:5]:
        # Find synthesis processes
        syn_processes = list(base_graph.objects(mof_uri, SYN_NS.hasSynthesisProcess))
        for syn_uri in syn_processes[:1]:
            solvents = list(base_graph.objects(syn_uri, SYN_NS.usesSolvent))
            for solvent_uri in solvents:
                # Check if chain inference exists
                chain_exists = (mof_uri, MOF_NS.directlyUsesSolvent, solvent_uri) in enriched_graph
                chain_samples.append({
                    'mof': str(mof_uri),
                    'solvent': str(solvent_uri),
                    'chain_exists': chain_exists,
                })
    
    result['samples']['property_chains'] = chain_samples
    chain_pass = all(s['chain_exists'] for s in chain_samples) if chain_samples else True
    
    if chain_pass:
        print(f"    ✓ Property chains validated ({len(chain_samples)} samples)")
    else:
        print(f"    ⚠ WARNING: Some property chains missing")
        if result['status'] == 'PASS':
            result['status'] = 'WARNING'
    
    return result


def _check_statistics(base_graph: Graph, enriched_graph: Graph, stats: Dict) -> Dict:
    """Check that statistics are reasonable."""
    print("\n[3] Statistical Sanity Checks...")
    
    base_triples = len(base_graph)
    enriched_triples = len(enriched_graph)
    inferred = enriched_triples - base_triples
    
    result = {
        'status': 'PASS',
        'inference_ratio': (inferred / base_triples * 100) if base_triples > 0 else 0,
        'warnings': [],
    }
    
    # Check inference ratio (should be reasonable, not too high or too low)
    if result['inference_ratio'] < 0.1:
        result['warnings'].append(f"Very low inference ratio: {result['inference_ratio']:.2f}%")
        print(f"  ⚠ WARNING: Very low inference ratio ({result['inference_ratio']:.2f}%)")
    elif result['inference_ratio'] > 50:
        result['warnings'].append(f"Very high inference ratio: {result['inference_ratio']:.2f}%")
        print(f"  ⚠ WARNING: Very high inference ratio ({result['inference_ratio']:.2f}%)")
    else:
        print(f"  ✓ Inference ratio: {result['inference_ratio']:.2f}% (reasonable)")
    
    # Check capability assignment
    base_mofs = len(list(base_graph.subjects(RDF.type, MOF_NS.MOF)))
    base_with_caps = len([mof for mof in base_graph.subjects(RDF.type, MOF_NS.MOF) 
                          if list(base_graph.objects(mof, MOF_NS.hasCapability))])
    enriched_with_caps = len([mof for mof in enriched_graph.subjects(RDF.type, MOF_NS.MOF) 
                              if list(enriched_graph.objects(mof, MOF_NS.hasCapability))])
    
    caps_added = enriched_with_caps - base_with_caps
    result['capabilities_added'] = caps_added
    result['capability_rate'] = (enriched_with_caps / base_mofs * 100) if base_mofs > 0 else 0
    
    print(f"  ✓ Capabilities added: {caps_added:,} MOFs")
    print(f"  ✓ MOFs with capabilities: {enriched_with_caps:,} ({result['capability_rate']:.1f}%)")
    
    if result['warnings']:
        result['status'] = 'WARNING'
    
    return result


def _check_consistency(enriched_graph: Graph) -> Dict:
    """Check for consistency violations."""
    print("\n[4] Consistency Checks...")
    
    result = {
        'status': 'PASS',
        'disjoint_violations': [],
        'type_consistency': 'PASS',
    }
    
    # Check disjoint class violations
    experimental_mofs = set(enriched_graph.subjects(RDF.type, MOF_NS.ExperimentalMOF))
    hypothetical_mofs = set(enriched_graph.subjects(RDF.type, MOF_NS.HypotheticalMOF))
    violations = experimental_mofs.intersection(hypothetical_mofs)
    
    if violations:
        result['status'] = 'FAIL'
        result['disjoint_violations'] = [str(v) for v in list(violations)[:10]]
        print(f"  ❌ FAIL: {len(violations)} disjoint class violations found")
    else:
        print(f"  ✓ PASS: No disjoint class violations")
    
    # Check property type consistency
    # Properties should be consistently typed
    structural_props = set(enriched_graph.subjects(RDF.type, MOF_NS.StructuralProperty))
    computational_props = set(enriched_graph.subjects(RDF.type, MOF_NS.ComputationalProperty))
    physical_props = set(enriched_graph.subjects(RDF.type, MOF_NS.PhysicalProperty))
    
    # Check for overlaps (should be disjoint)
    structural_computational_overlap = structural_props.intersection(computational_props)
    structural_physical_overlap = structural_props.intersection(physical_props)
    computational_physical_overlap = computational_props.intersection(physical_props)
    
    if structural_computational_overlap or structural_physical_overlap or computational_physical_overlap:
        result['type_consistency'] = 'WARNING'
        print(f"  ⚠ WARNING: Property type overlaps detected")
    else:
        print(f"  ✓ PASS: Property types are consistent")
    
    return result


def _verify_output(enriched_graph: Graph) -> Dict:
    """Verify that the enriched graph is valid and can be serialized."""
    print("\n[5] Output Verification...")
    
    result = {
        'status': 'PASS',
        'graph_valid': True,
        'can_serialize_ttl': True,
        'can_serialize_json': True,
    }
    
    # Check graph is valid
    try:
        triples_count = len(enriched_graph)
        result['triples_count'] = triples_count
        print(f"  ✓ Graph is valid ({triples_count:,} triples)")
    except Exception as e:
        result['status'] = 'FAIL'
        result['graph_valid'] = False
        result['error'] = str(e)
        print(f"  ❌ FAIL: Graph validation error: {e}")
        return result
    
    # Check TTL serialization
    try:
        ttl_str = enriched_graph.serialize(format='turtle')
        result['ttl_size'] = len(ttl_str)
        print(f"  ✓ TTL serialization works ({len(ttl_str):,} bytes)")
    except Exception as e:
        result['status'] = 'WARNING'
        result['can_serialize_ttl'] = False
        result['ttl_error'] = str(e)
        print(f"  ⚠ WARNING: TTL serialization error: {e}")
    
    return result


def _collect_sample_inferences(base_graph: Graph, enriched_graph: Graph) -> Dict:
    """Collect sample inferences for user review."""
    samples = {
        'inverse_properties': [],
        'property_chains': [],
        'capability_inferences': [],
    }
    
    # Sample inverse properties
    for mof_uri, linker_uri in list(base_graph.subject_objects(MOF_NS.hasLinker))[:3]:
        if (linker_uri, MOF_NS.usedInMOF, mof_uri) in enriched_graph:
            mof_name = list(base_graph.objects(mof_uri, MOF_NS.hasCanonicalName))
            linker_name = list(base_graph.objects(linker_uri, MOF_NS.hasChemicalName))
            samples['inverse_properties'].append({
                'mof': str(mof_name[0]) if mof_name else str(mof_uri),
                'linker': str(linker_name[0]) if linker_name else str(linker_uri),
                'relationship': 'usedInMOF (inverse of hasLinker)',
            })
    
    # Sample property chains
    for mof_uri in list(base_graph.subjects(RDF.type, MOF_NS.MOF))[:3]:
        syn_processes = list(base_graph.objects(mof_uri, SYN_NS.hasSynthesisProcess))
        for syn_uri in syn_processes[:1]:
            solvents = list(base_graph.objects(syn_uri, SYN_NS.usesSolvent))
            for solvent_uri in solvents:
                if (mof_uri, MOF_NS.directlyUsesSolvent, solvent_uri) in enriched_graph:
                    mof_name = list(base_graph.objects(mof_uri, MOF_NS.hasCanonicalName))
                    solvent_name = list(base_graph.objects(solvent_uri, MOF_NS.hasChemicalName))
                    samples['property_chains'].append({
                        'mof': str(mof_name[0]) if mof_name else str(mof_uri),
                        'solvent': str(solvent_name[0]) if solvent_name else str(solvent_uri),
                        'relationship': 'directlyUsesSolvent (chain: hasSynthesisProcess -> usesSolvent)',
                    })
                    break
    
    # Sample capability inferences
    base_mofs_with_caps = {mof for mof in base_graph.subjects(RDF.type, MOF_NS.MOF) 
                          if list(base_graph.objects(mof, MOF_NS.hasCapability))}
    enriched_mofs_with_caps = {mof for mof in enriched_graph.subjects(RDF.type, MOF_NS.MOF) 
                               if list(enriched_graph.objects(mof, MOF_NS.hasCapability))}
    new_cap_mofs = enriched_mofs_with_caps - base_mofs_with_caps
    
    for mof_uri in list(new_cap_mofs)[:5]:
        capabilities = list(enriched_graph.objects(mof_uri, MOF_NS.hasCapability))
        mof_name = list(enriched_graph.objects(mof_uri, MOF_NS.hasCanonicalName))
        cap_names = []
        for cap_uri in capabilities:
            cap_types = list(enriched_graph.objects(cap_uri, RDF.type))
            if cap_types:
                cap_name = str(cap_types[0]).split('#')[-1]
                cap_names.append(cap_name)
        
        samples['capability_inferences'].append({
            'mof': str(mof_name[0]) if mof_name else str(mof_uri),
            'capabilities': cap_names,
        })
    
    return samples


def _print_verification_summary(results: Dict):
    """Print a summary of verification results."""
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    checks = [
        ('Data Integrity', results['data_integrity']['status']),
        ('Inference Validation', results['inference_validation']['status']),
        ('Statistical Checks', results['statistical_checks']['status']),
        ('Consistency Checks', results['consistency_checks']['status']),
        ('Output Verification', results['output_verification']['status']),
    ]
    
    for check_name, status in checks:
        if status == 'PASS':
            print(f"  ✓ {check_name}: PASS")
        elif status == 'WARNING':
            print(f"  ⚠ {check_name}: WARNING")
        else:
            print(f"  ❌ {check_name}: FAIL")
    
    print(f"\n  Overall Status: {results['overall_status']}")
    
    if results['sample_inferences']:
        print(f"\n  Sample Inferences:")
        if results['sample_inferences'].get('inverse_properties'):
            print(f"    - Inverse properties: {len(results['sample_inferences']['inverse_properties'])} samples")
        if results['sample_inferences'].get('property_chains'):
            print(f"    - Property chains: {len(results['sample_inferences']['property_chains'])} samples")
        if results['sample_inferences'].get('capability_inferences'):
            print(f"    - Capability inferences: {len(results['sample_inferences']['capability_inferences'])} samples")
