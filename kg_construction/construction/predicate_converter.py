"""
Predicate to RDF Converter

Converts normalized predicate/relationship data to RDF triples.
Also handles implicit relationships from entity foreign keys.
"""

from rdflib import Graph, URIRef
from typing import Dict, List

from .namespace_manager import (
    get_mof_uri, get_linker_uri, get_cluster_uri, get_topology_uri,
    get_space_group_uri, get_crystal_system_uri, get_property_uri,
    get_synthesis_uri, get_condition_uri, get_abstract_uri,
    get_capability_uri, get_functionalized_mof_uri, get_functionalization_uri,
    get_chemical_uri, get_lattice_param_uri, get_solvent_uri, MOF_NS, SYN_NS
)


def add_linker_relation(graph: Graph, rel_data: Dict):
    """Add hasLinker relationship."""
    mof_id = rel_data['mof_id']
    linker_id = rel_data['linker_id']
    
    mof_uri = get_mof_uri(mof_id)
    linker_uri = get_linker_uri(linker_id)
    
    graph.add((mof_uri, MOF_NS.hasLinker, linker_uri))
    # Add inverse
    graph.add((linker_uri, MOF_NS.usedInMOF, mof_uri))


def add_metal_node_relation(graph: Graph, rel_data: Dict):
    """Add hasMetalNode relationship."""
    mof_id = rel_data['mof_id']
    # Handle both keys: 'cluster_id' (legacy) and 'metal_node_id' (from extractors)
    cluster_id = rel_data.get('cluster_id') or rel_data.get('metal_node_id')
    
    if not cluster_id:
        return
    
    mof_uri = get_mof_uri(mof_id)
    cluster_uri = get_cluster_uri(cluster_id)
    
    graph.add((mof_uri, MOF_NS.hasMetalNode, cluster_uri))
    # Add inverse
    graph.add((cluster_uri, MOF_NS.isComponentOf, mof_uri))


def add_topology_relation(graph: Graph, rel_data: Dict):
    """Add hasTopology relationship."""
    mof_id = rel_data['mof_id']
    topology_id = rel_data['topology_id']
    
    mof_uri = get_mof_uri(mof_id)
    topology_uri = get_topology_uri(topology_id)
    
    graph.add((mof_uri, MOF_NS.hasTopology, topology_uri))


def add_space_group_relation(graph: Graph, rel_data: Dict):
    """Add hasSpaceGroup relationship (via hasStructuralProperty)."""
    mof_id = rel_data['mof_id']
    sg_id = rel_data['space_group_id']
    
    mof_uri = get_mof_uri(mof_id)
    sg_uri = get_space_group_uri(sg_id)
    
    # Space groups are structural properties
    graph.add((mof_uri, MOF_NS.hasStructuralProperty, sg_uri))
    # Add inverse
    graph.add((sg_uri, MOF_NS.hasStructuralPropertyOwner, mof_uri))


def add_crystal_system_relation(graph: Graph, rel_data: Dict):
    """Add hasCrystalSystem relationship (via hasStructuralProperty)."""
    mof_id = rel_data['mof_id']
    cs_id = rel_data['crystal_system_id']
    
    mof_uri = get_mof_uri(mof_id)
    cs_uri = get_crystal_system_uri(cs_id)
    
    # Crystal systems are structural properties
    graph.add((mof_uri, MOF_NS.hasStructuralProperty, cs_uri))
    # Add inverse
    graph.add((cs_uri, MOF_NS.hasStructuralPropertyOwner, mof_uri))


def add_lattice_parameters_relation(graph: Graph, rel_data: Dict):
    """Add hasLatticeParameters relationship."""
    mof_id = rel_data['mof_id']
    lp_id = rel_data['lattice_param_id']
    
    mof_uri = get_mof_uri(mof_id)
    lp_uri = get_lattice_param_uri(lp_id)
    
    graph.add((mof_uri, MOF_NS.hasStructuralProperty, lp_uri))
    # Add inverse
    graph.add((lp_uri, MOF_NS.hasStructuralPropertyOwner, mof_uri))


def add_property_relation_from_entity(graph: Graph, prop_data: Dict):
    """Add property relationship from PropertyEntity (implicit relationship)."""
    mof_id = prop_data['mof_id']
    property_id = prop_data['property_id']
    property_type = prop_data.get('property_type', 'StructuralProperty')
    
    mof_uri = get_mof_uri(mof_id)
    prop_uri = get_property_uri(property_id)
    
    # Use specific property relationship based on type
    if property_type == 'StructuralProperty':
        graph.add((mof_uri, MOF_NS.hasStructuralProperty, prop_uri))
    elif property_type == 'ComputationalProperty':
        graph.add((mof_uri, MOF_NS.hasComputationalProperty, prop_uri))
    elif property_type == 'PhysicalProperty':
        graph.add((mof_uri, MOF_NS.hasPhysicalProperty, prop_uri))
    else:
        graph.add((mof_uri, MOF_NS.hasProperty, prop_uri))
    
    # Add inverse
    if property_type == 'StructuralProperty':
        graph.add((prop_uri, MOF_NS.hasStructuralPropertyOwner, mof_uri))
    elif property_type == 'ComputationalProperty':
        graph.add((prop_uri, MOF_NS.hasComputationalPropertyOwner, mof_uri))
    elif property_type == 'PhysicalProperty':
        graph.add((prop_uri, MOF_NS.hasPhysicalPropertyOwner, mof_uri))
    else:
        graph.add((prop_uri, MOF_NS.hasPropertyOwner, mof_uri))


def add_synthesis_process_relation_from_entity(graph: Graph, syn_data: Dict):
    """Add synthesis process relationship from SynthesisProcessEntity."""
    mof_id = syn_data['mof_id']
    synthesis_id = syn_data['synthesis_id']
    
    mof_uri = get_mof_uri(mof_id)
    syn_uri = get_synthesis_uri(synthesis_id)
    
    graph.add((mof_uri, SYN_NS.hasSynthesisProcess, syn_uri))
    # Add inverse
    graph.add((syn_uri, MOF_NS.hasMOF, mof_uri))


def add_condition_relation(graph: Graph, rel_data: Dict):
    """Add hasCondition relationship."""
    synthesis_id = rel_data['synthesis_id']
    condition_id = rel_data['condition_id']
    
    syn_uri = get_synthesis_uri(synthesis_id)
    cond_uri = get_condition_uri(condition_id)
    
    graph.add((syn_uri, SYN_NS.hasCondition, cond_uri))


def add_condition_relation_from_entity(graph: Graph, cond_data: Dict):
    """Add condition relationship from SynthesisConditionEntity."""
    synthesis_id = cond_data['synthesis_id']
    condition_id = cond_data['condition_id']
    
    syn_uri = get_synthesis_uri(synthesis_id)
    cond_uri = get_condition_uri(condition_id)
    
    graph.add((syn_uri, SYN_NS.hasCondition, cond_uri))


def add_abstract_relation(graph: Graph, rel_data: Dict):
    """Add hasAbstract relationship."""
    mof_id = rel_data['mof_id']
    abstract_id = rel_data['abstract_id']
    
    mof_uri = get_mof_uri(mof_id)
    abs_uri = get_abstract_uri(abstract_id)
    
    graph.add((mof_uri, MOF_NS.hasAbstract, abs_uri))
    # Add inverse
    graph.add((abs_uri, MOF_NS.describedInAbstract, mof_uri))


def add_abstract_relation_from_entity(graph: Graph, abs_data: Dict):
    """Add abstract relationship from AbstractEntity."""
    mof_id = abs_data['mof_id']
    abstract_id = abs_data['abstract_id']
    
    mof_uri = get_mof_uri(mof_id)
    abs_uri = get_abstract_uri(abstract_id)
    
    graph.add((mof_uri, MOF_NS.hasAbstract, abs_uri))
    graph.add((abs_uri, MOF_NS.describedInAbstract, mof_uri))


def add_capability_relation(graph: Graph, rel_data: Dict):
    """Add hasCapability relationship."""
    mof_id = rel_data['mof_id']
    capability_id = rel_data['capability_id']
    
    mof_uri = get_mof_uri(mof_id)
    cap_uri = get_capability_uri(capability_id)
    
    graph.add((mof_uri, MOF_NS.hasCapability, cap_uri))


def add_capability_relation_from_entity(graph: Graph, cap_data: Dict):
    """Add capability relationship from CapabilityEntity."""
    mof_id = cap_data['mof_id']
    capability_id = cap_data['capability_id']
    
    mof_uri = get_mof_uri(mof_id)
    cap_uri = get_capability_uri(capability_id)
    
    graph.add((mof_uri, MOF_NS.hasCapability, cap_uri))


def add_derived_from_relation(graph: Graph, rel_data: Dict):
    """Add derivedFrom relationship."""
    func_mof_id = rel_data['func_mof_id']
    parent_mof_id = rel_data['parent_mof_id']
    
    func_mof_uri = get_functionalized_mof_uri(func_mof_id)
    parent_mof_uri = get_mof_uri(parent_mof_id)
    
    graph.add((func_mof_uri, SYN_NS.derivedFrom, parent_mof_uri))


def add_functionalization_relation(graph: Graph, rel_data: Dict):
    """Add hasFunctionalization relationship."""
    # mof_id here refers to functionalized MOF (func_mof_id)
    mof_id = rel_data['mof_id']  # This is actually func_mof_id
    functionalization_id = rel_data['functionalization_id']
    
    # Check if it's a func_mof_id (contains underscore pattern) or regular mof_id
    # Functionalized MOF IDs typically have pattern like "KOJZEP_nmen"
    if '_' in mof_id and len(mof_id.split('_')) >= 2:
        func_mof_uri = get_functionalized_mof_uri(mof_id)
    else:
        # Fallback to regular MOF URI (shouldn't happen for functionalization relations)
        func_mof_uri = get_mof_uri(mof_id)
    
    func_uri = get_functionalization_uri(functionalization_id)
    
    graph.add((func_mof_uri, SYN_NS.hasFunctionalization, func_uri))


def add_functionalization_type_relation(graph: Graph, rel_data: Dict):
    """Add hasFunctionalizationType relationship."""
    functionalization_id = rel_data['functionalization_id']
    func_type_str = rel_data.get('functionalization_type', '')
    
    func_uri = get_functionalization_uri(functionalization_id)
    
    # Map string to ontology class
    func_type_map = {
        'AmineFunctionalization': SYN_NS.AmineFunctionalization,
        'MetalSubstitution': SYN_NS.MetalSubstitution,
        'LinkerModification': SYN_NS.LinkerModification,
        'Grafting': SYN_NS.Grafting,
        'MetalExchange': SYN_NS.MetalExchange,
    }
    
    if func_type_str in func_type_map:
        graph.add((func_uri, SYN_NS.hasFunctionalizationType, func_type_map[func_type_str]))


def add_uses_functional_group_relation(graph: Graph, rel_data: Dict):
    """Add usesFunctionalGroup relationship."""
    functionalization_id = rel_data['functionalization_id']
    # Handle both keys: 'functional_group_id' and 'chemical_id'
    functional_group_id = rel_data.get('functional_group_id') or rel_data.get('chemical_id')
    
    if not functional_group_id:
        return
    
    func_uri = get_functionalization_uri(functionalization_id)
    chem_uri = get_chemical_uri(functional_group_id)
    
    graph.add((func_uri, SYN_NS.usesFunctionalGroup, chem_uri))


def add_uses_solvent_relation(graph: Graph, rel_data: Dict):
    """Add usesSolvent relationship (SynthesisProcess → Solvent)."""
    synthesis_id = rel_data['synthesis_id']
    solvent_id = rel_data['solvent_id']
    
    syn_uri = get_synthesis_uri(synthesis_id)
    solvent_uri = get_solvent_uri(solvent_id)
    
    graph.add((syn_uri, SYN_NS.usesSolvent, solvent_uri))
    # Add inverse
    graph.add((solvent_uri, MOF_NS.usedInSynthesis, syn_uri))


def add_uses_additive_relation(graph: Graph, rel_data: Dict):
    """Add usesAdditive relationship (SynthesisProcess → Additive)."""
    synthesis_id = rel_data['synthesis_id']
    chemical_id = rel_data['chemical_id']
    
    syn_uri = get_synthesis_uri(synthesis_id)
    chem_uri = get_chemical_uri(chemical_id)
    
    graph.add((syn_uri, SYN_NS.usesAdditive, chem_uri))
    # Add inverse
    graph.add((chem_uri, MOF_NS.usedAsAdditiveIn, syn_uri))

