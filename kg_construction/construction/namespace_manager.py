"""
Namespace Manager for KG Construction

Defines and manages RDF namespaces for the MOF Knowledge Graph.
"""

from rdflib import Namespace, URIRef, Graph
from urllib.parse import quote
import re

# Define namespaces
MOF_NS = Namespace("http://emmo.info/domain-mof/mof-ontology#")
SYN_NS = Namespace("http://emmo.info/domain-mof/synthesis#")
EMMO_NS = Namespace("https://w3id.org/emmo#")


def sanitize_uri_component(component: str) -> str:
    """Sanitize a string to be used in a URI component."""
    # Replace invalid URI characters with underscores
    # Keep alphanumeric, underscore, hyphen, period
    # Replace spaces and invalid chars with underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', str(component))
    # URL encode special characters that might still be problematic
    return quote(sanitized, safe='_-.')

# Entity base URIs
def get_mof_uri(mof_id: str) -> URIRef:
    """Get URI for a MOF entity."""
    sanitized = sanitize_uri_component(mof_id)
    return MOF_NS[f"MOF_{sanitized}"]

def get_linker_uri(linker_id: str) -> URIRef:
    """Get URI for a linker entity."""
    sanitized = sanitize_uri_component(linker_id)
    return MOF_NS[sanitized]

def get_cluster_uri(cluster_id: str) -> URIRef:
    """Get URI for a metal cluster entity."""
    sanitized = sanitize_uri_component(cluster_id)
    return MOF_NS[sanitized]

def get_topology_uri(topology_id: str) -> URIRef:
    """Get URI for a topology entity."""
    sanitized = sanitize_uri_component(topology_id)
    return MOF_NS[f"Topology_{sanitized}"]

def get_space_group_uri(sg_id: str) -> URIRef:
    """Get URI for a space group entity."""
    sanitized = sanitize_uri_component(sg_id)
    return MOF_NS[f"SpaceGroup_{sanitized}"]

def get_crystal_system_uri(cs_id: str) -> URIRef:
    """Get URI for a crystal system entity."""
    sanitized = sanitize_uri_component(cs_id)
    return MOF_NS[f"CrystalSystem_{sanitized}"]

def get_property_uri(property_id: str) -> URIRef:
    """Get URI for a property entity."""
    sanitized = sanitize_uri_component(property_id)
    return MOF_NS[sanitized]

def get_synthesis_uri(synthesis_id: str) -> URIRef:
    """Get URI for a synthesis process entity."""
    sanitized = sanitize_uri_component(synthesis_id)
    return SYN_NS[sanitized]

def get_condition_uri(condition_id: str) -> URIRef:
    """Get URI for a synthesis condition entity."""
    sanitized = sanitize_uri_component(condition_id)
    return SYN_NS[sanitized]

def get_abstract_uri(abstract_id: str) -> URIRef:
    """Get URI for an abstract entity."""
    sanitized = sanitize_uri_component(abstract_id)
    return MOF_NS[sanitized]

def get_capability_uri(capability_id: str) -> URIRef:
    """Get URI for a capability entity."""
    sanitized = sanitize_uri_component(capability_id)
    return MOF_NS[sanitized]

def get_functionalized_mof_uri(func_mof_id: str) -> URIRef:
    """Get URI for a functionalized MOF entity."""
    sanitized = sanitize_uri_component(func_mof_id)
    return MOF_NS[f"FuncMOF_{sanitized}"]

def get_functionalization_uri(functionalization_id: str) -> URIRef:
    """Get URI for a functionalization entity."""
    sanitized = sanitize_uri_component(functionalization_id)
    return SYN_NS[sanitized]

def get_solvent_uri(solvent_id: str) -> URIRef:
    """Get URI for a solvent entity."""
    sanitized = sanitize_uri_component(solvent_id)
    return SYN_NS[f"Solvent_{sanitized}"]

def get_chemical_uri(chemical_id: str) -> URIRef:
    """Get URI for a chemical entity."""
    sanitized = sanitize_uri_component(chemical_id)
    return MOF_NS[sanitized]

def get_lattice_param_uri(lattice_param_id: str) -> URIRef:
    """Get URI for a lattice parameter entity."""
    sanitized = sanitize_uri_component(lattice_param_id)
    return MOF_NS[sanitized]

def setup_namespaces(graph: Graph):
    """Bind namespaces to graph for cleaner output."""
    graph.bind("", MOF_NS)
    graph.bind("syn", SYN_NS)
    graph.bind("emmo", EMMO_NS)
    graph.bind("owl", "http://www.w3.org/2002/07/owl#")
    graph.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
    graph.bind("xsd", "http://www.w3.org/2001/XMLSchema#")
