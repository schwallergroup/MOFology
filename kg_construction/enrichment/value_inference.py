"""
Value-Based Property Inference

Applies value-based thresholds to infer capabilities from property values.
This complements ontology-based reasoning with programmatic value checks.
"""

from typing import Dict, List, Tuple
from rdflib import Graph, RDF, Literal, URIRef, Namespace

import sys
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from construction.namespace_manager import MOF_NS, SYN_NS


# Property value thresholds for capability inference
THRESHOLDS = {
    'co2_uptake': {
        'property_names': ['CO2 uptake', 'CO2 uptake (mmol/g)', 'CO2 uptake (wt%)'],
        'min_value': 2.0,  # mmol/g
        'capability': MOF_NS.CO2CaptureCapability,
        'units': ['mmol/g', 'mmol g-1', 'mmol/g MOF'],
    },
    'co2_binding_energy': {
        'property_names': ['CO2 binding energy', 'Binding Energy CO2', 'CO2 binding energy (kJ/mol)'],
        'min_value': -60.0,  # kJ/mol (more negative = stronger binding)
        'max_value': -30.0,
        'capability': MOF_NS.CO2CaptureCapability,
        'units': ['kJ/mol', 'kJ mol-1', 'eV'],
    },
    'co2_low_pressure': {
        'property_names': ['CO2 uptake at LP', 'CO2 uptake at low pressure', 'CO2 uptake (400 ppm)'],
        'min_value': 0.5,  # mmol/g at low pressure
        'capability': MOF_NS.DACCapability,
        'units': ['mmol/g', 'mmol g-1'],
    },
    'band_gap': {
        'property_names': ['Band gap', 'Band gap (eV)', 'Bandgap'],
        'min_value': 1.5,  # eV
        'max_value': 3.5,
        'capability': MOF_NS.PhotocatalyticCapability,
        'units': ['eV', 'electron volt'],
    },
    'h2_uptake': {
        'property_names': ['H2 uptake', 'H2 uptake (wt%)', 'H2 storage capacity'],
        'min_value': 5.5,  # wt%
        'capability': MOF_NS.HydrogenStorageCapability,
        'units': ['wt%', 'weight percent', '%'],
    },
    'ch4_uptake': {
        'property_names': ['CH4 uptake', 'CH4 uptake (cm³/g)', 'CH4 storage capacity'],
        'min_value': 200.0,  # cm³(STP)/g
        'capability': MOF_NS.MethaneStorageCapability,
        'units': ['cm³/g', 'cm3/g', 'cm³(STP)/g'],
    },
}


def apply_value_based_inference(graph: Graph) -> Tuple[Graph, Dict]:
    """
    Apply value-based property thresholds to infer capabilities.
    
    Args:
        graph: The RDF graph (already enriched with OWL-RL reasoning)
        
    Returns:
        Tuple of (enriched_graph, statistics_dict)
    """
    print("\n" + "=" * 80)
    print("APPLYING VALUE-BASED INFERENCE")
    print("=" * 80)
    
    stats = {
        'value_based_capabilities_added': 0,
        'capabilities_by_type': {},
        'properties_checked': 0,
        'thresholds_applied': {},
    }
    
    # Track which MOFs already have capabilities (to avoid duplicates)
    mofs_with_existing_capabilities = {}
    for mof_uri in graph.subjects(RDF.type, MOF_NS.MOF):
        existing_caps = list(graph.objects(mof_uri, MOF_NS.hasCapability))
        if existing_caps:
            mofs_with_existing_capabilities[mof_uri] = set(existing_caps)
        else:
            mofs_with_existing_capabilities[mof_uri] = set()
    
    # Process each threshold rule
    for threshold_name, threshold_config in THRESHOLDS.items():
        print(f"\n[{threshold_name}] Checking {threshold_config['property_names'][0]}...")
        
        capability = threshold_config['capability']
        capability_name = str(capability).split('#')[-1] if '#' in str(capability) else str(capability)
        
        if capability_name not in stats['capabilities_by_type']:
            stats['capabilities_by_type'][capability_name] = 0
        
        properties_checked = 0
        capabilities_added = 0
        
        # Find all properties matching the names
        for prop_uri in graph.subjects(RDF.type, MOF_NS.MaterialProperty):
            prop_name_lit = list(graph.objects(prop_uri, MOF_NS.propertyName))
            if not prop_name_lit:
                continue
            
            prop_name = str(prop_name_lit[0]).lower()
            
            # Check if property name matches
            matches = False
            for pattern in threshold_config['property_names']:
                if pattern.lower() in prop_name:
                    matches = True
                    break
            
            if not matches:
                continue
            
            # Get property value and units
            prop_value_lit = list(graph.objects(prop_uri, MOF_NS.propertyValue))
            prop_units_lit = list(graph.objects(prop_uri, MOF_NS.propertyUnits))
            
            if not prop_value_lit:
                continue
            
            try:
                prop_value = float(prop_value_lit[0])
                prop_units = str(prop_units_lit[0]).lower() if prop_units_lit else ""
                
                properties_checked += 1
                
                # Check units (if specified)
                if threshold_config.get('units'):
                    units_match = any(unit.lower() in prop_units for unit in threshold_config['units'])
                    if not units_match and prop_units:  # If units are specified but don't match, skip
                        continue
                
                # Apply threshold check
                passes_threshold = False
                
                if 'min_value' in threshold_config and 'max_value' in threshold_config:
                    passes_threshold = (threshold_config['min_value'] <= prop_value <= threshold_config['max_value'])
                elif 'min_value' in threshold_config:
                    passes_threshold = (prop_value >= threshold_config['min_value'])
                elif 'max_value' in threshold_config:
                    passes_threshold = (prop_value <= threshold_config['max_value'])
                
                if passes_threshold:
                    # Find MOF(s) that have this property
                    mof_uris = []
                    
                    # Check direct property relationships
                    for mof_uri in graph.subjects(MOF_NS.hasProperty, prop_uri):
                        mof_uris.append(mof_uri)
                    for mof_uri in graph.subjects(MOF_NS.hasStructuralProperty, prop_uri):
                        mof_uris.append(mof_uri)
                    for mof_uri in graph.subjects(MOF_NS.hasComputationalProperty, prop_uri):
                        mof_uris.append(mof_uri)
                    for mof_uri in graph.subjects(MOF_NS.hasPhysicalProperty, prop_uri):
                        mof_uris.append(mof_uri)
                    
                    # Check inverse relationships
                    for mof_uri in graph.objects(prop_uri, MOF_NS.hasPropertyOwner):
                        mof_uris.append(mof_uri)
                    for mof_uri in graph.objects(prop_uri, MOF_NS.hasStructuralPropertyOwner):
                        mof_uris.append(mof_uri)
                    for mof_uri in graph.objects(prop_uri, MOF_NS.hasComputationalPropertyOwner):
                        mof_uris.append(mof_uri)
                    for mof_uri in graph.objects(prop_uri, MOF_NS.hasPhysicalPropertyOwner):
                        mof_uris.append(mof_uri)
                    
                    # Add capability to MOF if not already present
                    for mof_uri in set(mof_uris):
                        if capability not in mofs_with_existing_capabilities[mof_uri]:
                            graph.add((mof_uri, MOF_NS.hasCapability, capability))
                            mofs_with_existing_capabilities[mof_uri].add(capability)
                            capabilities_added += 1
                            stats['value_based_capabilities_added'] += 1
                            stats['capabilities_by_type'][capability_name] += 1
            except (ValueError, TypeError):
                # Skip properties with invalid values
                continue
        
        stats['thresholds_applied'][threshold_name] = {
            'properties_checked': properties_checked,
            'capabilities_added': capabilities_added,
        }
        stats['properties_checked'] += properties_checked
        
        print(f"    Properties checked: {properties_checked:,}")
        print(f"    Capabilities added: {capabilities_added:,}")
    
    print(f"\n    Total value-based capabilities added: {stats['value_based_capabilities_added']:,}")
    
    return graph, stats
