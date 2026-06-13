"""
Entity to RDF Converter

Converts normalized entity data to RDF triples according to the ontology.
"""

from rdflib import Graph, Literal, URIRef, XSD, RDF, RDFS
from typing import Dict, List
import json

from .namespace_manager import (
    get_mof_uri, get_linker_uri, get_cluster_uri, get_topology_uri,
    get_space_group_uri, get_crystal_system_uri, get_property_uri,
    get_synthesis_uri, get_condition_uri, get_abstract_uri,
    get_capability_uri, get_functionalized_mof_uri, get_functionalization_uri,
    get_chemical_uri, get_lattice_param_uri, get_solvent_uri, MOF_NS, SYN_NS
)


def add_mof_entity(graph: Graph, mof_data: Dict) -> URIRef:
    """Add MOF entity to graph."""
    mof_id = mof_data['mof_id']
    mof_uri = get_mof_uri(mof_id)
    
    # Determine MOF type
    if mof_data.get('is_experimental', True):
        mof_class = MOF_NS.ExperimentalMOF
    else:
        mof_class = MOF_NS.HypotheticalMOF
    
    graph.add((mof_uri, RDF.type, MOF_NS.MOF))
    graph.add((mof_uri, RDF.type, mof_class))
    
    # Datatype properties
    if mof_data.get('canonical_name'):
        graph.add((mof_uri, MOF_NS.hasCanonicalName, Literal(mof_data['canonical_name'], datatype=XSD.string)))
    
    if mof_data.get('formula'):
        graph.add((mof_uri, MOF_NS.hasFormula, Literal(mof_data['formula'], datatype=XSD.string)))
    
    if mof_data.get('csd_code'):
        graph.add((mof_uri, MOF_NS.hasCSDCode, Literal(mof_data['csd_code'], datatype=XSD.string)))
    
    if mof_data.get('mp_id'):
        graph.add((mof_uri, MOF_NS.hasMPID, Literal(mof_data['mp_id'], datatype=XSD.string)))
    
    if mof_data.get('mofid'):
        graph.add((mof_uri, MOF_NS.hasMOFid, Literal(mof_data['mofid'], datatype=XSD.string)))
    
    # Alternative names
    for name in mof_data.get('all_names', []):
        if name and name != mof_data.get('canonical_name'):
            graph.add((mof_uri, MOF_NS.hasAlternativeName, Literal(name, datatype=XSD.string)))
    
    # isHypothetical (inverted from is_experimental)
    is_hypothetical = not mof_data.get('is_experimental', True)
    graph.add((mof_uri, SYN_NS.isHypothetical, Literal(is_hypothetical, datatype=XSD.boolean)))
    
    return mof_uri


def add_linker_entity(graph: Graph, linker_data: Dict) -> URIRef:
    """Add linker entity to graph."""
    linker_id = linker_data['linker_id']
    linker_uri = get_linker_uri(linker_id)
    
    graph.add((linker_uri, RDF.type, MOF_NS.OrganicLinker))
    
    if linker_data.get('canonical_name'):
        graph.add((linker_uri, MOF_NS.hasChemicalName, Literal(linker_data['canonical_name'], datatype=XSD.string)))
    
    if linker_data.get('smiles'):
        graph.add((linker_uri, MOF_NS.hasSMILES, Literal(linker_data['smiles'], datatype=XSD.string)))
    
    if linker_data.get('canonical_smiles'):
        graph.add((linker_uri, MOF_NS.hasCanonicalSMILES, Literal(linker_data['canonical_smiles'], datatype=XSD.string)))
    
    for name in linker_data.get('all_names', []):
        if name and name != linker_data.get('canonical_name'):
            graph.add((linker_uri, MOF_NS.hasAlternativeChemicalName, Literal(name, datatype=XSD.string)))
    
    return linker_uri


def add_metal_cluster_entity(graph: Graph, cluster_data: Dict) -> URIRef:
    """Add metal cluster entity to graph."""
    cluster_id = cluster_data['cluster_id']
    cluster_uri = get_cluster_uri(cluster_id)
    
    graph.add((cluster_uri, RDF.type, MOF_NS.MetalCluster))
    
    for element in cluster_data.get('metal_elements', []):
        graph.add((cluster_uri, MOF_NS.hasMetalElement, Literal(element, datatype=XSD.string)))
    
    if cluster_data.get('formula'):
        graph.add((cluster_uri, MOF_NS.hasClusterFormula, Literal(cluster_data['formula'], datatype=XSD.string)))
    
    if cluster_data.get('description'):
        graph.add((cluster_uri, MOF_NS.hasClusterDescription, Literal(cluster_data['description'], datatype=XSD.string)))
    
    if cluster_data.get('coordination_number') is not None:
        graph.add((cluster_uri, MOF_NS.coordinationNumber, Literal(cluster_data['coordination_number'], datatype=XSD.integer)))
    
    return cluster_uri


def add_topology_entity(graph: Graph, topology_data: Dict) -> URIRef:
    """Add topology entity to graph."""
    topology_id = topology_data['topology_id']
    topology_uri = get_topology_uri(topology_id)
    
    graph.add((topology_uri, RDF.type, MOF_NS.Topology))
    
    if topology_data.get('topology_name'):
        graph.add((topology_uri, MOF_NS.topologyCode, Literal(topology_data['topology_name'], datatype=XSD.string)))
    
    return topology_uri


def add_space_group_entity(graph: Graph, sg_data: Dict) -> URIRef:
    """Add space group entity to graph."""
    sg_id = sg_data['space_group_id']
    sg_uri = get_space_group_uri(sg_id)
    
    # Space groups are StructuralProperty entities
    graph.add((sg_uri, RDF.type, MOF_NS.StructuralProperty))
    
    if sg_data.get('space_group_name'):
        graph.add((sg_uri, MOF_NS.propertyName, Literal(f"Space group: {sg_data['space_group_name']}", datatype=XSD.string)))
    
    if sg_data.get('space_group_number') is not None:
        graph.add((sg_uri, MOF_NS.propertyValue, Literal(sg_data['space_group_number'], datatype=XSD.decimal)))
    
    return sg_uri


def add_crystal_system_entity(graph: Graph, cs_data: Dict) -> URIRef:
    """Add crystal system entity to graph."""
    cs_id = cs_data['crystal_system_id']
    cs_uri = get_crystal_system_uri(cs_id)
    
    # Crystal systems are StructuralProperty entities
    graph.add((cs_uri, RDF.type, MOF_NS.StructuralProperty))
    
    if cs_data.get('crystal_system_name'):
        graph.add((cs_uri, MOF_NS.propertyName, Literal(f"Crystal system: {cs_data['crystal_system_name']}", datatype=XSD.string)))
    
    return cs_uri


def add_property_entity(graph: Graph, prop_data: Dict) -> URIRef:
    """Add property entity to graph."""
    property_id = prop_data['property_id']
    property_uri = get_property_uri(property_id)
    
    # Determine property type
    prop_type = prop_data.get('property_type', 'StructuralProperty')
    
    # Map property types to ontology classes
    if prop_type == 'StructuralProperty':
        prop_class = MOF_NS.StructuralProperty
    elif prop_type == 'ComputationalProperty':
        prop_class = MOF_NS.ComputationalProperty
    elif prop_type == 'PhysicalProperty':
        prop_class = MOF_NS.PhysicalProperty
    elif prop_type == 'SurfaceAreaProperty':
        prop_class = MOF_NS.SurfaceAreaProperty
    elif prop_type == 'PoreVolumeProperty':
        prop_class = MOF_NS.PoreVolumeProperty
    elif prop_type == 'DensityProperty':
        prop_class = MOF_NS.DensityProperty
    elif prop_type == 'PoreSizeProperty':
        prop_class = MOF_NS.PoreSizeProperty
    elif prop_type == 'BandGapProperty':
        prop_class = MOF_NS.BandGapProperty
    elif prop_type == 'BindingEnergyProperty':
        prop_class = MOF_NS.BindingEnergyProperty
    elif prop_type == 'HenryCoefficientProperty':
        prop_class = MOF_NS.HenryCoefficientProperty
    elif prop_type == 'GasUptakeProperty':
        prop_class = MOF_NS.GasUptakeProperty
    elif prop_type == 'VoidFractionProperty':
        prop_class = MOF_NS.VoidFractionProperty
    elif prop_type == 'ElectronicProperty':
        prop_class = MOF_NS.ElectronicProperty
    elif prop_type == 'AdsorptionProperty':
        prop_class = MOF_NS.AdsorptionProperty
    elif prop_type == 'FreeEnergy':
        prop_class = MOF_NS.FreeEnergy
    elif prop_type == 'StrainEnergy':
        prop_class = MOF_NS.StrainEnergy
    else:
        # Fallback or generic MaterialProperty
        # Try to use the prop_type as the class name if it looks like one
        if 'Property' in prop_type:
            prop_class = MOF_NS[prop_type]
        else:
            prop_class = MOF_NS.MaterialProperty
    
    graph.add((property_uri, RDF.type, MOF_NS.MaterialProperty))
    graph.add((property_uri, RDF.type, prop_class))
    
    if prop_data.get('property_name'):
        graph.add((property_uri, MOF_NS.propertyName, Literal(prop_data['property_name'], datatype=XSD.string)))
    
    if prop_data.get('value') is not None:
        graph.add((property_uri, MOF_NS.propertyValue, Literal(float(prop_data['value']), datatype=XSD.decimal)))
    
    if prop_data.get('units'):
        graph.add((property_uri, MOF_NS.propertyUnits, Literal(prop_data['units'], datatype=XSD.string)))
    
    if prop_data.get('conditions'):
        graph.add((property_uri, MOF_NS.propertyConditions, Literal(prop_data['conditions'], datatype=XSD.string)))
    
    return property_uri


def add_synthesis_process_entity(graph: Graph, syn_data: Dict) -> URIRef:
    """Add synthesis process entity to graph."""
    synthesis_id = syn_data['synthesis_id']
    syn_uri = get_synthesis_uri(synthesis_id)
    
    graph.add((syn_uri, RDF.type, SYN_NS.SynthesisProcess))
    
    if syn_data.get('method'):
        graph.add((syn_uri, SYN_NS.hasSynthesisMethod, Literal(syn_data['method'], datatype=XSD.string)))
    
    if syn_data.get('yield_percent') is not None:
        graph.add((syn_uri, SYN_NS.hasYield, Literal(float(syn_data['yield_percent']), datatype=XSD.decimal)))
    
    # Counterions (repeatable datatype property)
    for counterion in syn_data.get('counterions', []):
        if counterion:
            graph.add((syn_uri, SYN_NS.hasCounterion, Literal(counterion, datatype=XSD.string)))
    
    return syn_uri


def add_synthesis_condition_entity(graph: Graph, cond_data: Dict) -> URIRef:
    """Add synthesis condition entity to graph."""
    condition_id = cond_data['condition_id']
    cond_uri = get_condition_uri(condition_id)
    
    graph.add((cond_uri, RDF.type, SYN_NS.SynthesisCondition))
    
    if cond_data.get('temperature_c') is not None:
        graph.add((cond_uri, SYN_NS.hasTemperature, Literal(float(cond_data['temperature_c']), datatype=XSD.decimal)))
    
    if cond_data.get('pressure_bar') is not None:
        graph.add((cond_uri, SYN_NS.hasPressure, Literal(float(cond_data['pressure_bar']), datatype=XSD.decimal)))
    
    if cond_data.get('time_hours') is not None:
        graph.add((cond_uri, SYN_NS.hasReactionTime, Literal(float(cond_data['time_hours']), datatype=XSD.decimal)))
    
    return cond_uri


def add_abstract_entity(graph: Graph, abs_data: Dict) -> URIRef:
    """Add abstract entity to graph."""
    abstract_id = abs_data['abstract_id']
    abs_uri = get_abstract_uri(abstract_id)
    
    graph.add((abs_uri, RDF.type, MOF_NS.Abstract))
    graph.add((abs_uri, RDF.type, MOF_NS.Publication))
    
    if abs_data.get('title'):
        graph.add((abs_uri, MOF_NS.publicationTitle, Literal(abs_data['title'], datatype=XSD.string)))
    
    if abs_data.get('abstract_text'):
        graph.add((abs_uri, MOF_NS.publicationAbstract, Literal(abs_data['abstract_text'], datatype=XSD.string)))
    
    if abs_data.get('authors'):
        graph.add((abs_uri, MOF_NS.publicationAuthors, Literal(abs_data['authors'], datatype=XSD.string)))
    
    if abs_data.get('journal'):
        graph.add((abs_uri, MOF_NS.publicationJournal, Literal(abs_data['journal'], datatype=XSD.string)))
    
    if abs_data.get('doi'):
        # Extract proper DOI from file path if needed
        doi = abs_data['doi']
        doi_str = str(doi)
        
        # If DOI is stored as file path, extract the actual DOI (format: 10.XXXX/...)
        if '10.' in doi_str:
            idx = doi_str.find('10.')
            doi_str = doi_str[idx:]
            # Clean up any URL encoding
            doi_str = doi_str.replace('%2F', '/')
        
        graph.add((abs_uri, MOF_NS.publicationDOI, Literal(doi_str, datatype=XSD.string)))
    
    return abs_uri


def add_capability_entity(graph: Graph, cap_data: Dict) -> URIRef:
    """Add capability entity to graph."""
    capability_id = cap_data['capability_id']
    cap_uri = get_capability_uri(capability_id)
    
    # Determine capability type
    cap_type = cap_data.get('capability_type', 'Capability')
    if cap_type == 'CO2CaptureCapability':
        cap_class = MOF_NS.CO2CaptureCapability
    elif cap_type == 'HydrogenStorageCapability':
        cap_class = MOF_NS.HydrogenStorageCapability
    elif cap_type == 'MethaneStorageCapability':
        cap_class = MOF_NS.MethaneStorageCapability
    elif cap_type == 'PhotocatalyticCapability':
        cap_class = MOF_NS.PhotocatalyticCapability
    elif cap_type == 'LuminescentSensingCapability':
        cap_class = MOF_NS.LuminescentSensingCapability
    elif cap_type == 'DACCapability':
        cap_class = MOF_NS.DACCapability
    elif cap_type == 'CatalysisCapability':
        cap_class = MOF_NS.CatalysisCapability
    else:
        cap_class = MOF_NS.Capability
    
    graph.add((cap_uri, RDF.type, MOF_NS.Capability))
    graph.add((cap_uri, RDF.type, cap_class))
    
    if cap_data.get('value') is not None:
        graph.add((cap_uri, MOF_NS.hasValue, Literal(float(cap_data['value']), datatype=XSD.decimal)))
    
    return cap_uri


def add_functionalized_mof_entity(graph: Graph, func_mof_data: Dict) -> URIRef:
    """Add functionalized MOF entity to graph."""
    func_mof_id = func_mof_data['func_mof_id']
    func_mof_uri = get_functionalized_mof_uri(func_mof_id)
    
    graph.add((func_mof_uri, RDF.type, MOF_NS.MOF))
    graph.add((func_mof_uri, RDF.type, SYN_NS.FunctionalizedMOF))
    graph.add((func_mof_uri, RDF.type, MOF_NS.ExperimentalMOF))
    
    if func_mof_data.get('canonical_name'):
        graph.add((func_mof_uri, MOF_NS.hasCanonicalName, Literal(func_mof_data['canonical_name'], datatype=XSD.string)))
    
    if func_mof_data.get('formula'):
        graph.add((func_mof_uri, MOF_NS.hasFormula, Literal(func_mof_data['formula'], datatype=XSD.string)))
    
    return func_mof_uri


def add_functionalization_entity(graph: Graph, func_data: Dict) -> URIRef:
    """Add functionalization entity to graph."""
    func_id = func_data['functionalization_id']
    func_uri = get_functionalization_uri(func_id)
    
    graph.add((func_uri, RDF.type, SYN_NS.Functionalization))
    
    if func_data.get('functionalization_method'):
        graph.add((func_uri, SYN_NS.functionalizationMethod, Literal(func_data['functionalization_method'], datatype=XSD.string)))
    
    if func_data.get('functionalization_degree') is not None:
        graph.add((func_uri, SYN_NS.functionalizationDegree, Literal(float(func_data['functionalization_degree']), datatype=XSD.decimal)))
    
    if func_data.get('functional_group_name'):
        graph.add((func_uri, SYN_NS.functionalGroupName, Literal(func_data['functional_group_name'], datatype=XSD.string)))
    
    if func_data.get('functional_group_smiles'):
        graph.add((func_uri, SYN_NS.functionalGroupSMILES, Literal(func_data['functional_group_smiles'], datatype=XSD.string)))
    
    return func_uri


def add_chemical_entity(graph: Graph, chem_data: Dict) -> URIRef:
    """Add chemical entity to graph."""
    chemical_id = chem_data['chemical_id']
    chem_uri = get_chemical_uri(chemical_id)
    
    graph.add((chem_uri, RDF.type, MOF_NS.Chemical))
    
    if chem_data.get('canonical_name'):
        graph.add((chem_uri, MOF_NS.hasChemicalName, Literal(chem_data['canonical_name'], datatype=XSD.string)))
    
    if chem_data.get('smiles'):
        graph.add((chem_uri, MOF_NS.hasSMILES, Literal(chem_data['smiles'], datatype=XSD.string)))
    
    if chem_data.get('canonical_smiles'):
        graph.add((chem_uri, MOF_NS.hasCanonicalSMILES, Literal(chem_data['canonical_smiles'], datatype=XSD.string)))
    
    for name in chem_data.get('all_names', []):
        if name and name != chem_data.get('canonical_name'):
            graph.add((chem_uri, MOF_NS.hasAlternativeChemicalName, Literal(name, datatype=XSD.string)))
    
    return chem_uri


def add_lattice_parameter_entity(graph: Graph, lp_data: Dict) -> URIRef:
    """Add lattice parameter entity to graph."""
    lp_id = lp_data['lattice_param_id']
    lp_uri = get_lattice_param_uri(lp_id)
    
    graph.add((lp_uri, RDF.type, MOF_NS.StructuralProperty))
    
    if lp_data.get('property_name'):
        graph.add((lp_uri, MOF_NS.propertyName, Literal(lp_data['property_name'], datatype=XSD.string)))
    
    if lp_data.get('units'):
        graph.add((lp_uri, MOF_NS.propertyUnits, Literal(lp_data['units'], datatype=XSD.string)))
    
    # Store individual parameters (could also store as structured data)
    if lp_data.get('volume') is not None:
        graph.add((lp_uri, MOF_NS.propertyValue, Literal(float(lp_data['volume']), datatype=XSD.decimal)))
    
    return lp_uri


def add_solvent_entity(graph: Graph, solvent_data: Dict) -> URIRef:
    """Add solvent entity to graph."""
    solvent_id = solvent_data['solvent_id']
    solvent_uri = get_solvent_uri(solvent_id)
    
    graph.add((solvent_uri, RDF.type, MOF_NS.Chemical))
    graph.add((solvent_uri, RDF.type, SYN_NS.Solvent))
    
    if solvent_data.get('canonical_name'):
        graph.add((solvent_uri, MOF_NS.hasChemicalName, Literal(solvent_data['canonical_name'], datatype=XSD.string)))
    
    if solvent_data.get('smiles'):
        graph.add((solvent_uri, MOF_NS.hasSMILES, Literal(solvent_data['smiles'], datatype=XSD.string)))
    
    if solvent_data.get('canonical_smiles'):
        graph.add((solvent_uri, MOF_NS.hasCanonicalSMILES, Literal(solvent_data['canonical_smiles'], datatype=XSD.string)))
    
    for name in solvent_data.get('all_names', []):
        if name and name != solvent_data.get('canonical_name'):
            graph.add((solvent_uri, MOF_NS.hasAlternativeChemicalName, Literal(name, datatype=XSD.string)))
    
    return solvent_uri


def add_additive_entity(graph: Graph, chem_data: Dict) -> URIRef:
    """Add additive entity to graph (typed as syn:Additive)."""
    chemical_id = chem_data['chemical_id']
    chem_uri = get_chemical_uri(chemical_id)
    
    graph.add((chem_uri, RDF.type, MOF_NS.Chemical))
    graph.add((chem_uri, RDF.type, SYN_NS.Additive))
    
    if chem_data.get('canonical_name'):
        graph.add((chem_uri, MOF_NS.hasChemicalName, Literal(chem_data['canonical_name'], datatype=XSD.string)))
    
    if chem_data.get('smiles'):
        graph.add((chem_uri, MOF_NS.hasSMILES, Literal(chem_data['smiles'], datatype=XSD.string)))
    
    if chem_data.get('canonical_smiles'):
        graph.add((chem_uri, MOF_NS.hasCanonicalSMILES, Literal(chem_data['canonical_smiles'], datatype=XSD.string)))
    
    for name in chem_data.get('all_names', []):
        if name and name != chem_data.get('canonical_name'):
            graph.add((chem_uri, MOF_NS.hasAlternativeChemicalName, Literal(name, datatype=XSD.string)))
    
    return chem_uri




