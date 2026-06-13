"""
Main Knowledge Graph Builder

Orchestrates the construction of the RDF knowledge graph from normalized entities and predicates.
"""

import json
import re
from pathlib import Path
from typing import Dict, List
from rdflib import Graph, RDF, Namespace, URIRef, Literal, BNode

from .namespace_manager import setup_namespaces, get_mof_uri, get_functionalized_mof_uri, MOF_NS, SYN_NS
from .entity_converter import (
    add_mof_entity, add_linker_entity, add_metal_cluster_entity, add_topology_entity,
    add_space_group_entity, add_crystal_system_entity, add_property_entity,
    add_synthesis_process_entity, add_synthesis_condition_entity, add_abstract_entity,
    add_capability_entity, add_functionalized_mof_entity, add_functionalization_entity,
    add_chemical_entity, add_lattice_parameter_entity, add_solvent_entity, add_additive_entity
)
from .predicate_converter import (
    add_linker_relation, add_metal_node_relation, add_topology_relation,
    add_space_group_relation, add_crystal_system_relation, add_lattice_parameters_relation,
    add_property_relation_from_entity, add_synthesis_process_relation_from_entity,
    add_condition_relation, add_condition_relation_from_entity,
    add_abstract_relation, add_abstract_relation_from_entity,
    add_capability_relation, add_capability_relation_from_entity,
    add_derived_from_relation, add_functionalization_relation,
    add_functionalization_type_relation, add_uses_functional_group_relation,
    add_uses_solvent_relation, add_uses_additive_relation
)


class KGBuilder:
    """Builds RDF knowledge graph from normalized data."""
    
    def __init__(self, normalized_data_dir: Path, ontology_file: Path):
        self.normalized_data_dir = Path(normalized_data_dir)
        self.ontology_file = Path(ontology_file)
        self.graph = Graph()
        setup_namespaces(self.graph)
        
        # Statistics
        self.stats = {
            'entities': {},
            'relationships': {},
            'triples': 0
        }
    
    def load_ontology(self):
        """Load ontology into graph."""
        if self.ontology_file.exists():
            self.graph.parse(str(self.ontology_file), format='turtle')
            print(f"Loaded ontology from {self.ontology_file}")
        else:
            print(f"Warning: Ontology file not found: {self.ontology_file}")
    
    def load_and_add_entities(self):
        """Load all entity files and add to graph."""
        print("\nLoading entities...")
        
        # MOFs
        self._load_entities_file("normalized_mofs.json", add_mof_entity, "MOFs")
        
        # Linkers
        self._load_entities_file("normalized_linkers.json", add_linker_entity, "Linkers")
        
        # Metal Clusters
        self._load_entities_file("normalized_metal_clusters.json", add_metal_cluster_entity, "Metal Clusters")
        
        # Topologies
        self._load_entities_file("normalized_topologies.json", add_topology_entity, "Topologies")
        
        # Space Groups
        self._load_entities_file("normalized_space_groups.json", add_space_group_entity, "Space Groups")
        
        # Crystal Systems
        self._load_entities_file("normalized_crystal_systems.json", add_crystal_system_entity, "Crystal Systems")
        
        # Properties
        self._load_entities_file("normalized_properties.json", add_property_entity, "Properties")
        
        # Lattice Parameters
        self._load_entities_file("normalized_lattice_parameters.json", add_lattice_parameter_entity, "Lattice Parameters")
        
        # Synthesis Processes
        self._load_entities_file("normalized_synthesis_processes.json", add_synthesis_process_entity, "Synthesis Processes")
        
        # Synthesis Conditions
        self._load_entities_file("normalized_synthesis_conditions.json", add_synthesis_condition_entity, "Synthesis Conditions")
        
        # Abstracts
        self._load_entities_file("normalized_abstracts.json", add_abstract_entity, "Abstracts")
        
        # Capabilities
        self._load_entities_file("normalized_capabilities.json", add_capability_entity, "Capabilities")
        
        # Functionalized MOFs
        self._load_entities_file("normalized_functionalized_mofs.json", add_functionalized_mof_entity, "Functionalized MOFs")
        
        # Functionalizations
        self._load_entities_file("normalized_functionalizations.json", add_functionalization_entity, "Functionalizations")
        
        # Chemicals
        self._load_entities_file("normalized_chemicals.json", add_chemical_entity, "Chemicals")
        
        # Solvents
        self._load_entities_file("normalized_solvents.json", add_solvent_entity, "Solvents")
        
        # Additives (typed as syn:Additive in the KG)
        self._load_entities_file("normalized_additives.json", add_additive_entity, "Additives")
    
    def _load_entities_file(self, filename: str, converter_func, entity_type: str):
        """Helper to load and convert entities from a file."""
        filepath = self.normalized_data_dir / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found")
            return
        
        with open(filepath) as f:
            entities = json.load(f)
        
        count = 0
        for entity_data in entities:
            try:
                converter_func(self.graph, entity_data)
                count += 1
            except Exception as e:
                print(f"  Error processing {entity_type}: {e}")
        
        self.stats['entities'][entity_type] = count
        print(f"  Added {count:,} {entity_type}")
    
    def add_relationships_from_entities(self):
        """Add implicit relationships from entity foreign keys."""
        print("\nAdding relationships from entities...")
        
        # Properties → MOFs
        props_file = self.normalized_data_dir / "normalized_properties.json"
        if props_file.exists():
            with open(props_file) as f:
                properties = json.load(f)
            
            count = 0
            for prop in properties:
                try:
                    add_property_relation_from_entity(self.graph, prop)
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['hasProperty (from entities)'] = count
            print(f"  Added {count:,} property relationships")
        
        # Synthesis Processes → MOFs
        syns_file = self.normalized_data_dir / "normalized_synthesis_processes.json"
        if syns_file.exists():
            with open(syns_file) as f:
                syntheses = json.load(f)
            
            count = 0
            for syn in syntheses:
                try:
                    add_synthesis_process_relation_from_entity(self.graph, syn)
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['hasSynthesisProcess (from entities)'] = count
            print(f"  Added {count:,} synthesis process relationships")
        
        # Synthesis Conditions → Synthesis Processes
        conds_file = self.normalized_data_dir / "normalized_synthesis_conditions.json"
        if conds_file.exists():
            with open(conds_file) as f:
                conditions = json.load(f)
            
            count = 0
            for cond in conditions:
                try:
                    add_condition_relation_from_entity(self.graph, cond)
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['hasCondition (from entities)'] = count
            print(f"  Added {count:,} synthesis condition relationships")
        
        # Abstracts → MOFs
        abs_file = self.normalized_data_dir / "normalized_abstracts.json"
        if abs_file.exists():
            with open(abs_file) as f:
                abstracts = json.load(f)
            
            count = 0
            for abstract in abstracts:
                try:
                    add_abstract_relation_from_entity(self.graph, abstract)
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['hasAbstract (from entities)'] = count
            print(f"  Added {count:,} abstract relationships")
        
        # Capabilities → MOFs
        caps_file = self.normalized_data_dir / "normalized_capabilities.json"
        if caps_file.exists():
            with open(caps_file) as f:
                capabilities = json.load(f)
            
            count = 0
            for cap in capabilities:
                try:
                    add_capability_relation_from_entity(self.graph, cap)
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['hasCapability (from entities)'] = count
            print(f"  Added {count:,} capability relationships")
        
        # Lattice Parameters → MOFs
        lps_file = self.normalized_data_dir / "normalized_lattice_parameters.json"
        if lps_file.exists():
            with open(lps_file) as f:
                lattice_params = json.load(f)
            
            count = 0
            for lp in lattice_params:
                try:
                    add_lattice_parameters_relation(self.graph, {'mof_id': lp['mof_id'], 'lattice_param_id': lp['lattice_param_id']})
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['hasLatticeParameters (from entities)'] = count
            print(f"  Added {count:,} lattice parameter relationships")
        
        # Functionalized MOFs → Parent MOFs
        func_mofs_file = self.normalized_data_dir / "normalized_functionalized_mofs.json"
        if func_mofs_file.exists():
            with open(func_mofs_file) as f:
                func_mofs = json.load(f)
            
            count = 0
            for func_mof in func_mofs:
                try:
                    add_derived_from_relation(self.graph, {
                        'func_mof_id': func_mof['func_mof_id'],
                        'parent_mof_id': func_mof['parent_csd_code']
                    })
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['derivedFrom (from entities)'] = count
            print(f"  Added {count:,} derivedFrom relationships")
        
        # Functionalizations → Functionalized MOFs
        funcs_file = self.normalized_data_dir / "normalized_functionalizations.json"
        if funcs_file.exists():
            with open(funcs_file) as f:
                funcs = json.load(f)
            
            count = 0
            for func in funcs:
                try:
                    add_functionalization_relation(self.graph, {
                        'mof_id': func['func_mof_id'],
                        'functionalization_id': func['functionalization_id']
                    })
                    count += 1
                except Exception as e:
                    pass
            self.stats['relationships']['hasFunctionalization (from entities)'] = count
            print(f"  Added {count:,} functionalization relationships")
    
    def load_and_add_predicates(self):
        """Load predicate files and add relationships to graph."""
        print("\nLoading predicates...")
        
        predicate_handlers = {
            "normalized_has_linker_relations.json": add_linker_relation,
            "normalized_has_metal_node_relations.json": add_metal_node_relation,
            "normalized_has_topology_relations.json": add_topology_relation,
            "normalized_has_space_group_relations.json": add_space_group_relation,
            "normalized_has_crystal_system_relations.json": add_crystal_system_relation,
            "normalized_has_lattice_parameters_relations.json": add_lattice_parameters_relation,
            "normalized_has_synthesis_process_relations.json": lambda g, d: None,  # Already handled from entities
            "normalized_has_condition_relations.json": add_condition_relation,
            "normalized_has_abstract_relations.json": lambda g, d: None,  # Already handled from entities
            "normalized_has_capability_relations.json": lambda g, d: None,  # Already handled from entities
            "normalized_derived_from_relations.json": add_derived_from_relation,
            "normalized_has_functionalization_relations.json": add_functionalization_relation,
            "normalized_has_functionalization_type_relations.json": add_functionalization_type_relation,
            "normalized_uses_functional_group_relations.json": add_uses_functional_group_relation,
            "normalized_uses_solvent_relations.json": add_uses_solvent_relation,
            "normalized_uses_additive_relations.json": add_uses_additive_relation,
        }
        
        for filename, handler in predicate_handlers.items():
            filepath = self.normalized_data_dir / filename
            if not filepath.exists():
                continue
            
            with open(filepath) as f:
                predicates = json.load(f)
            
            count = 0
            for pred_data in predicates:
                try:
                    handler(self.graph, pred_data)
                    count += 1
                except Exception as e:
                    pass
            
            # Only count if handler actually does something
            if handler.__name__ != '<lambda>':
                rel_name = filename.replace('normalized_', '').replace('_relations.json', '').replace('_', ' ')
                self.stats['relationships'][rel_name] = count
                print(f"  Added {count:,} {rel_name}")
    
    def save_graph(self, output_dir: Path):
        """Save graph as TTL and JSON-LD."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Turtle
        ttl_file = output_dir / "mof_kg.ttl"
        try:
            self.graph.serialize(destination=str(ttl_file), format='turtle')
            print(f"\nSaved KG as Turtle: {ttl_file}")
        except Exception as e:
            print(f"Error saving TTL: {e}")
        
        # Save as human-readable JSON
        json_file = output_dir / "mof_kg.json"
        try:
            self._save_human_readable_json(json_file)
            print(f"Saved KG as human-readable JSON: {json_file}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
        
        # Update stats
        self.stats['triples'] = len(self.graph)
    
    def _get_node_name(self, uri: URIRef) -> str:
        """Get human-readable name for a node URI."""
        # Try various name properties
        name_properties = [
            MOF_NS.hasCanonicalName,
            MOF_NS.hasChemicalName,
            MOF_NS.propertyName,
            MOF_NS.publicationTitle,
            MOF_NS.topologyCode,
        ]
        
        for prop in name_properties:
            names = list(self.graph.objects(uri, prop))
            if names:
                return str(names[0])
        
        # Fallback: extract from URI
        uri_str = str(uri)
        if '#' in uri_str:
            return uri_str.split('#')[-1]
        elif '/' in uri_str:
            return uri_str.split('/')[-1]
        return uri_str
    
    def _get_predicate_name(self, predicate: URIRef) -> str:
        """Get human-readable name for a predicate URI."""
        uri_str = str(predicate)
        if '#' in uri_str:
            name = uri_str.split('#')[-1]
        elif '/' in uri_str:
            name = uri_str.split('/')[-1]
        else:
            return uri_str
        
        # Convert camelCase to readable format
        # Insert space before capital letters
        name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        return name
    
    def _save_human_readable_json(self, json_file: Path):
        """Save graph as human-readable JSON while preserving node identity by URI."""
        # Build display-name mappings for URI/BNode nodes
        node_names = {}
        for term in set(list(self.graph.subjects()) + list(self.graph.objects())):
            if isinstance(term, URIRef):
                node_names[term] = self._get_node_name(term)
            elif isinstance(term, BNode):
                node_names[term] = f"_:{str(term)}"

        # Build predicate name mappings
        predicate_names = {}
        for p in self.graph.predicates():
            if isinstance(p, URIRef) and p != RDF.type:
                predicate_names[p] = self._get_predicate_name(p)

        # Build nodes keyed by stable ID (URI string or _:bnode id) to avoid accidental merges.
        nodes = {}
        for term, display_name in node_names.items():
            term_id = str(term) if isinstance(term, URIRef) else f"_:{str(term)}"
            nodes[term_id] = {
                'id': term_id,
                'uri': str(term) if isinstance(term, URIRef) else None,
                'name': display_name,
                'node_kind': 'uri' if isinstance(term, URIRef) else 'bnode',
                'types': []
            }

        # Populate rdf:type values.
        for s, _, o in self.graph.triples((None, RDF.type, None)):
            if isinstance(s, (URIRef, BNode)):
                sid = str(s) if isinstance(s, URIRef) else f"_:{str(s)}"
                if sid not in nodes:
                    if isinstance(s, URIRef):
                        nodes[sid] = {
                            'id': sid, 'uri': str(s), 'name': self._get_node_name(s),
                            'node_kind': 'uri', 'types': []
                        }
                    else:
                        nodes[sid] = {
                            'id': sid, 'uri': None, 'name': f"_:{str(s)}",
                            'node_kind': 'bnode', 'types': []
                        }
                type_name = str(o).split('#')[-1] if '#' in str(o) else str(o).split('/')[-1]
                if type_name not in nodes[sid]['types']:
                    nodes[sid]['types'].append(type_name)

        # Build edges preserving URI identity and include readable labels.
        edges = []
        for s, p, o in self.graph:
            if p == RDF.type:
                continue

            sid = str(s) if isinstance(s, URIRef) else f"_:{str(s)}"
            edge = {
                'subject_uri': sid,
                'subject_name': node_names.get(s, str(s)),
                'predicate_uri': str(p),
                'predicate': predicate_names.get(p, str(p)),
                'object': None
            }

            if isinstance(o, URIRef):
                edge['object'] = {
                    'object_uri': str(o),
                    'object_name': node_names.get(o, str(o)),
                    'object_kind': 'uri'
                }
            elif isinstance(o, BNode):
                edge['object'] = {
                    'object_uri': f"_:{str(o)}",
                    'object_name': node_names.get(o, f"_:{str(o)}"),
                    'object_kind': 'bnode'
                }
            elif isinstance(o, Literal):
                edge['object'] = {
                    'object_kind': 'literal',
                    'value': str(o),
                    'datatype': str(o.datatype) if o.datatype else None,
                    'language': o.language if o.language else None
                }
            else:
                edge['object'] = {
                    'object_kind': 'other',
                    'value': str(o)
                }

            edges.append(edge)

        json_data = {
            'nodes': list(nodes.values()),
            'edges': edges,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'total_triples': len(self.graph)
            }
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def get_statistics(self) -> Dict:
        """Get KG statistics."""
        # Count node types (entities by rdf:type)
        from rdflib import RDF
        node_types = {}
        for s, p, o in self.graph.triples((None, RDF.type, None)):
            node_type = str(o).split('#')[-1] if '#' in str(o) else str(o).split('/')[-1]
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count edge types (predicates)
        edge_types = {}
        for s, p, o in self.graph:
            if p != RDF.type:  # Exclude type assertions
                edge_name = str(p).split('#')[-1] if '#' in str(p) else str(p).split('/')[-1]
                edge_types[edge_name] = edge_types.get(edge_name, 0) + 1
        
        return {
            'total_triples': len(self.graph),
            'node_types': node_types,
            'edge_types': edge_types,
            'entities': self.stats['entities'],
            'relationships': self.stats['relationships']
        }

