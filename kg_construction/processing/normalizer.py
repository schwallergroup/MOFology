"""
Main Normalization Script

Orchestrates the cleaning and normalization of all entities.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

from ..datamodels.entitymodels import (
    MOFEntity, LinkerEntity, PropertyEntity, MetalClusterEntity,
    SynthesisProcessEntity, AbstractEntity, FunctionalizedMOFEntity,
    FunctionalizationEntity, ChemicalEntity, TopologyEntity,
    SpaceGroupEntity, CrystalSystemEntity, LatticeParameterEntity,
    CapabilityEntity, SynthesisConditionEntity, SolventEntity
)
from .linker_resolution import LinkerResolver

class Normalizer:
    """Orchestrates normalization and relationship resolution."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory stores
        self.mofs: Dict[str, MOFEntity] = {}
        self.linkers: Dict[str, LinkerEntity] = {}
        self.properties: List[PropertyEntity] = []
        self.metal_clusters: List[MetalClusterEntity] = []
        self.topologies: List[TopologyEntity] = []
        self.space_groups: List[SpaceGroupEntity] = []
        self.crystal_systems: List[CrystalSystemEntity] = []
        self.lattice_parameters: List[LatticeParameterEntity] = []
        self.synthesis_processes: List[SynthesisProcessEntity] = []
        self.synthesis_conditions: List[SynthesisConditionEntity] = []
        self.abstracts: List[AbstractEntity] = []
        self.capabilities: List[CapabilityEntity] = []
        self.functionalized_mofs: List[FunctionalizedMOFEntity] = []
        self.functionalizations: List[FunctionalizationEntity] = []
        self.chemicals: List[ChemicalEntity] = []
        self.solvents: List[SolventEntity] = []
        self.additives: List[ChemicalEntity] = []
        
        # Store relationships (predicate lists)
        self.relationships = {
            'has_linker': [], # {'mof_id': ..., 'linker_id': ...}
            'has_metal_node': [],
            'has_topology': [],
            'has_space_group': [],
            'has_crystal_system': [],
            'has_lattice_parameters': [],
            'has_condition': [],
            'derived_from': [],
            'has_functionalization': [],
            'has_functionalization_type': [],
            'uses_functional_group': [],
            'uses_solvent': [],
            'uses_additive': [],
        }

        self.linker_resolver = None

    def load_entities(self, entity_list: List[Any], entity_type: str):
        """Load raw entities into memory."""
        print(f"Loading {len(entity_list)} {entity_type}s...")
        if entity_type == 'MOF':
            for mof in entity_list:
                if mof.mof_id not in self.mofs:
                    self.mofs[mof.mof_id] = mof
                else:
                    # Merge logic
                    existing = self.mofs[mof.mof_id]
                    existing.all_names = list(set(existing.all_names + mof.all_names))
                    existing.data_sources = list(set(existing.data_sources + mof.data_sources))
                    if not existing.formula and mof.formula: existing.formula = mof.formula
                    if not existing.topology and mof.topology: existing.topology = mof.topology
                    # Merge other_ids
                    if mof.other_ids:
                        existing.other_ids.update(mof.other_ids)
        elif entity_type == 'Linker':
            for linker in entity_list:
                if linker.linker_id not in self.linkers:
                    self.linkers[linker.linker_id] = linker
        elif entity_type == 'Property':
            self.properties.extend(entity_list)
        elif entity_type == 'MetalCluster':
            self.metal_clusters.extend(entity_list)
        elif entity_type == 'Topology':
            self.topologies.extend(entity_list)
        elif entity_type == 'SpaceGroup':
            self.space_groups.extend(entity_list)
        elif entity_type == 'CrystalSystem':
            self.crystal_systems.extend(entity_list)
        elif entity_type == 'LatticeParameter':
            self.lattice_parameters.extend(entity_list)
        elif entity_type == 'SynthesisProcess':
            self.synthesis_processes.extend(entity_list)
        elif entity_type == 'SynthesisCondition':
            self.synthesis_conditions.extend(entity_list)
        elif entity_type == 'Abstract':
            self.abstracts.extend(entity_list)
        elif entity_type == 'Capability':
            self.capabilities.extend(entity_list)
        elif entity_type == 'FunctionalizedMOF':
            self.functionalized_mofs.extend(entity_list)
        elif entity_type == 'Functionalization':
            self.functionalizations.extend(entity_list)
        elif entity_type == 'Chemical':
            self.chemicals.extend(entity_list)
        elif entity_type == 'Solvent':
            self.solvents.extend(entity_list)
        elif entity_type == 'Additive':
            self.additives.extend(entity_list)
        else:
            print(f"Warning: Unknown entity type {entity_type}")

    def load_relationships(self, rel_list: List[Dict], rel_type: str):
        """Load explicit relationships."""
        if rel_type in self.relationships:
            self.relationships[rel_type].extend(rel_list)
        else:
            print(f"Warning: Unknown relationship type {rel_type}")

    def resolve_relationships(self):
        """Resolve cross-entity relationships."""
        print("Resolving relationships...")
        
        # 1. Deduplicate Metal Clusters, Topologies, etc. (simple)
        # (Assuming extractors did basic deduplication, but we can do more here if needed)
        
        # 2. Resolve Linker Relationships
        self.linker_resolver = LinkerResolver(list(self.linkers.values()))
        
        # If we have explicit has_linker relationships from LinkerExtractor, use them
        # (This is handled by load_relationships('has_linker'))
        
        # 3. Resolve Relationships from Entity Fields
        
        # MOF -> Topology
        for mof_id, mof in self.mofs.items():
            if mof.topology:
                # Assuming topology field matches a topology_id
                self.relationships['has_topology'].append({
                    'mof_id': mof_id,
                    'topology_id': mof.topology.lower()
                })
        
        # MOF -> Linker (if stored in mof.linker_ids)
        # Note: We rely on explicit relationships loaded from LinkerExtractor for now.
        
        # 4. Resolve Relationships from Foreign Keys in Entities
        # has_lattice_parameters
        for lp in self.lattice_parameters:
            if lp.mof_id:
                self.relationships['has_lattice_parameters'].append({
                    'mof_id': lp.mof_id,
                    'lattice_param_id': lp.lattice_param_id
                })
        
        # has_condition (from SynthesisCondition to SynthesisProcess)
        for cond in self.synthesis_conditions:
            if cond.synthesis_id:
                self.relationships['has_condition'].append({
                    'synthesis_id': cond.synthesis_id,
                    'condition_id': cond.condition_id
                })
        
        # derived_from (FunctionalizedMOF -> Parent MOF)
        for func_mof in self.functionalized_mofs:
            if func_mof.parent_csd_code:
                self.relationships['derived_from'].append({
                    'func_mof_id': func_mof.func_mof_id,
                    'parent_mof_id': func_mof.parent_csd_code
                })
        
        # has_functionalization (Functionalization -> FunctionalizedMOF)
        for func in self.functionalizations:
            if func.func_mof_id:
                self.relationships['has_functionalization'].append({
                    'mof_id': func.func_mof_id,
                    'functionalization_id': func.functionalization_id
                })
        
        # has_functionalization_type (Functionalization -> FunctionalizationType)
        for func in self.functionalizations:
            if func.functionalization_type:
                self.relationships['has_functionalization_type'].append({
                    'functionalization_id': func.functionalization_id,
                    'functionalization_type': func.functionalization_type
                })
                
        # uses_functional_group (Functionalization -> Chemical)
        for func in self.functionalizations:
            if func.functional_group_id:
                self.relationships['uses_functional_group'].append({
                    'functionalization_id': func.functionalization_id,
                    'chemical_id': func.functional_group_id
                })
        
        # Deduplicate relationships
        for key in self.relationships:
            # Convert list of dicts to list of tuples for set conversion, then back
            unique = {tuple(sorted(d.items())) for d in self.relationships[key]}
            self.relationships[key] = [dict(t) for t in unique]
            print(f"  {key}: {len(self.relationships[key])} relationships")

    def save_normalized(self):
        """Save normalized entities and relationships to JSON."""
        print(f"Saving normalized data to {self.output_dir}...")
        
        # Save Entities
        self._save_list(self.mofs.values(), 'normalized_mofs.json')
        self._save_list(self.linkers.values(), 'normalized_linkers.json')
        self._save_list(self.properties, 'normalized_properties.json')
        self._save_list(self.metal_clusters, 'normalized_metal_clusters.json')
        self._save_list(self.topologies, 'normalized_topologies.json')
        self._save_list(self.space_groups, 'normalized_space_groups.json')
        self._save_list(self.crystal_systems, 'normalized_crystal_systems.json')
        self._save_list(self.lattice_parameters, 'normalized_lattice_parameters.json')
        self._save_list(self.synthesis_processes, 'normalized_synthesis_processes.json')
        self._save_list(self.synthesis_conditions, 'normalized_synthesis_conditions.json')
        self._save_list(self.abstracts, 'normalized_abstracts.json')
        self._save_list(self.capabilities, 'normalized_capabilities.json')
        self._save_list(self.functionalized_mofs, 'normalized_functionalized_mofs.json')
        self._save_list(self.functionalizations, 'normalized_functionalizations.json')
        self._save_list(self.chemicals, 'normalized_chemicals.json')
        self._save_list(self.solvents, 'normalized_solvents.json')
        self._save_list(self.additives, 'normalized_additives.json')
        
        # Save Relationships (Predicates)
        for rel_type, rels in self.relationships.items():
            filename = f"normalized_{rel_type}_relations.json"
            with open(self.output_dir / filename, 'w') as f:
                json.dump(rels, f, indent=2)
        
        print("Normalization complete.")

    def _save_list(self, entities, filename):
        """Helper to save a list of entities."""
        with open(self.output_dir / filename, 'w') as f:
            json.dump([vars(e) for e in entities], f, indent=2, default=str)
