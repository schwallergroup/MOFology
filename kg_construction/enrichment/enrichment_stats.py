"""
Enrichment Statistics Tracking

Tracks and reports statistics about the enrichment process, including
inferred triples by type and capability assignments.
"""

from typing import Dict, List
from rdflib import Graph, RDF, Namespace
from collections import defaultdict, Counter

import sys
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from construction.namespace_manager import MOF_NS, SYN_NS


class EnrichmentStats:
    """Tracks enrichment statistics."""
    
    def __init__(self):
        self.base_stats = {}
        self.enriched_stats = {}
        self.inference_stats = {}
        self.capability_stats = {}
        self.value_based_stats = {}
    
    def collect_base_stats(self, graph: Graph):
        """Collect statistics from the base (un-enriched) graph."""
        print("\n" + "=" * 80)
        print("COLLECTING BASE STATISTICS")
        print("=" * 80)
        
        self.base_stats = {
            'total_triples': len(graph),
            'total_mofs': len(list(graph.subjects(RDF.type, MOF_NS.MOF))),
            'total_properties': len(list(graph.subjects(RDF.type, MOF_NS.MaterialProperty))),
            'total_linkers': len(list(graph.subjects(RDF.type, MOF_NS.OrganicLinker))),
            'total_capabilities': len(list(graph.subjects(RDF.type, MOF_NS.Capability))),
            'mofs_with_capabilities': len([mof for mof in graph.subjects(RDF.type, MOF_NS.MOF) 
                                          if list(graph.objects(mof, MOF_NS.hasCapability))]),
        }
        
        print(f"  Total triples: {self.base_stats['total_triples']:,}")
        print(f"  Total MOFs: {self.base_stats['total_mofs']:,}")
        print(f"  Total properties: {self.base_stats['total_properties']:,}")
        print(f"  MOFs with capabilities: {self.base_stats['mofs_with_capabilities']:,}")
    
    def collect_enriched_stats(self, graph: Graph, owlrl_stats: Dict, value_stats: Dict):
        """Collect statistics from the enriched graph."""
        print("\n" + "=" * 80)
        print("COLLECTING ENRICHED STATISTICS")
        print("=" * 80)
        
        self.enriched_stats = {
            'total_triples': len(graph),
            'total_mofs': len(list(graph.subjects(RDF.type, MOF_NS.MOF))),
            'total_properties': len(list(graph.subjects(RDF.type, MOF_NS.MaterialProperty))),
            'total_linkers': len(list(graph.subjects(RDF.type, MOF_NS.OrganicLinker))),
            'total_capabilities': len(list(graph.subjects(RDF.type, MOF_NS.Capability))),
            'mofs_with_capabilities': len([mof for mof in graph.subjects(RDF.type, MOF_NS.MOF) 
                                          if list(graph.objects(mof, MOF_NS.hasCapability))]),
        }
        
        # Store inference statistics
        self.inference_stats = {
            'total_inferred_triples': owlrl_stats.get('inferred_triples', 0),
            'inverse_property_inferences': owlrl_stats.get('inverse_property_inferences', 0),
            'property_chain_inferences': owlrl_stats.get('property_chain_inferences', 0),
            'transitive_property_inferences': owlrl_stats.get('transitive_property_inferences', 0),
            'ontology_capability_inferences': owlrl_stats.get('capability_inferences', 0),
            'disjoint_class_violations': len(owlrl_stats.get('disjoint_class_violations', [])),
        }
        
        # Store value-based statistics
        self.value_based_stats = value_stats
        
        # Calculate capability statistics
        self._collect_capability_stats(graph)
        
        print(f"  Total triples: {self.enriched_stats['total_triples']:,}")
        print(f"  Inferred triples: {self.inference_stats['total_inferred_triples']:,}")
        print(f"  MOFs with capabilities: {self.enriched_stats['mofs_with_capabilities']:,}")
        print(f"  Value-based capabilities added: {self.value_based_stats.get('value_based_capabilities_added', 0):,}")
    
    def _collect_capability_stats(self, graph: Graph):
        """Collect detailed capability statistics."""
        capability_counts = Counter()
        mof_capability_counts = defaultdict(int)
        
        for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
            capabilities = list(graph.objects(mof_uri, MOF_NS.hasCapability))
            mof_capability_counts[len(capabilities)] += 1
            
            for cap_uri in capabilities:
                cap_types = list(graph.objects(cap_uri, RDF.type))
                for cap_type in cap_types:
                    cap_name = str(cap_type).split('#')[-1] if '#' in str(cap_type) else str(cap_type)
                    capability_counts[cap_name] += 1
        
        self.capability_stats = {
            'capability_distribution': dict(capability_counts),
            'mofs_by_capability_count': dict(mof_capability_counts),
            'total_capability_assignments': sum(capability_counts.values()),
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics as a dictionary."""
        return {
            'base_kg': self.base_stats,
            'enriched_kg': self.enriched_stats,
            'inferences': self.inference_stats,
            'value_based': self.value_based_stats,
            'capabilities': self.capability_stats,
            'summary': {
                'triples_added': self.enriched_stats['total_triples'] - self.base_stats['total_triples'],
                'capabilities_added': self.enriched_stats['mofs_with_capabilities'] - self.base_stats['mofs_with_capabilities'],
                'inference_ratio': (self.inference_stats['total_inferred_triples'] / self.base_stats['total_triples'] * 100) 
                                  if self.base_stats['total_triples'] > 0 else 0,
            }
        }
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 80)
        print("ENRICHMENT SUMMARY")
        print("=" * 80)
        
        print(f"\nBase KG:")
        print(f"  Triples: {self.base_stats['total_triples']:,}")
        print(f"  MOFs: {self.base_stats['total_mofs']:,}")
        print(f"  MOFs with capabilities: {self.base_stats['mofs_with_capabilities']:,}")
        
        print(f"\nEnriched KG:")
        print(f"  Triples: {self.enriched_stats['total_triples']:,}")
        print(f"  MOFs: {self.enriched_stats['total_mofs']:,}")
        print(f"  MOFs with capabilities: {self.enriched_stats['mofs_with_capabilities']:,}")
        
        print(f"\nInferences:")
        print(f"  Total inferred triples: {self.inference_stats['total_inferred_triples']:,}")
        print(f"  Inverse property inferences: {self.inference_stats['inverse_property_inferences']:,}")
        print(f"  Property chain inferences: {self.inference_stats['property_chain_inferences']:,}")
        print(f"  Transitive property inferences: {self.inference_stats['transitive_property_inferences']:,}")
        print(f"  Ontology capability inferences: {self.inference_stats['ontology_capability_inferences']:,}")
        print(f"  Value-based capabilities added: {self.value_based_stats.get('value_based_capabilities_added', 0):,}")
        
        if self.capability_stats:
            print(f"\nCapability Distribution:")
            for cap_name, count in sorted(self.capability_stats['capability_distribution'].items(), 
                                        key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cap_name}: {count:,}")
