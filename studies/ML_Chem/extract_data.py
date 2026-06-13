import pandas as pd
from rdflib import Graph, Namespace, URIRef
import os
import sys
from collections import defaultdict

# Paths - use environment variable or default to local
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_PATH = f"{_BASE}/KG/data/KG/mof_kg.ttl"
OUTPUT_DIR = f"{_BASE}/studies/data/"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "chemcial_properties.csv")

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #load KG
    if not os.path.exists(KG_PATH):
        print(f"Error: KG file not found at {KG_PATH}")
        sys.exit(1)
    


    print(f"Loading KG from {KG_PATH} (this might take a minute)...")
    g = Graph()
    g.parse(KG_PATH, format="ttl")
    print("KG loaded successfully.")

    #Broad Sweep for Numerical Properties
    query_numeric = """
    PREFIX : <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?name ?value
    WHERE {
        { ?prop :hasComputationalPropertyOwner ?mof }
        UNION { ?prop :hasPhysicalPropertyOwner ?mof }
        UNION { ?prop :hasStructuralPropertyOwner ?mof }
        UNION { ?prop :hasPropertyOwner ?mof }
        ?prop :propertyName ?name ;
              :propertyValue ?value .
    }
    """
    print("Broad sweep for numerical properties...")
    results_numeric = g.query(query_numeric)
    print(f"Numerical records found: {len(results_numeric)}")



    #Broad Sweep for Chemical/Structural Attributes
    query_attributes = """
    PREFIX : <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?formula ?mofid ?topo ?metal ?smiles ?sg ?cs
    WHERE {
        { ?mof :hasFormula ?formula }
        UNION { ?mof :hasMOFid ?mofid }
        UNION { ?mof :hasTopology ?t . ?t :topologyCode ?topo }
        UNION { ?mof :hasMetalNode ?m . ?m :hasMetalElement ?metal }
        UNION { ?mof :hasLinker ?l . ?l :hasSMILES ?smiles }
        UNION { 
            ?mof :hasStructuralProperty ?sg_res . 
            FILTER(STRSTARTS(STR(?sg_res), "http://emmo.info/domain-mof/mof-ontology#SpaceGroup_"))
            BIND(STRAFTER(STR(?sg_res), "http://emmo.info/domain-mof/mof-ontology#SpaceGroup_") AS ?sg)
        }
        UNION { 
            ?mof :hasStructuralProperty ?cs_res . 
            FILTER(STRSTARTS(STR(?cs_res), "http://emmo.info/domain-mof/mof-ontology#CrystalSystem_"))
            BIND(STRAFTER(STR(?cs_res), "http://emmo.info/domain-mof/mof-ontology#CrystalSystem_") AS ?cs)
        }
    }
    """
    print("Broad sweep for chemical/structural attributes...")
    results_attr = g.query(query_attributes)
    print(f"Attribute records found: {len(results_attr)}")



   # Process results
    print("Processing results for all MOFs in KG...")
    
    # Numerical data - process ALL MOFs
    numeric_data = []
    for row in results_numeric:
        m = str(row.mof)
        try:
            numeric_data.append({
                'mof_uri': m,
                'property_name': str(row.name),
                'property_value': float(row.value)
            })
        except (ValueError, TypeError):
            continue
    
    # Attribute data - process ALL MOFs
    attr_collector = defaultdict(lambda: {
        'formula': set(), 'mofid': set(), 'topo': set(), 
        'metal': set(), 'smiles': set(), 'sg': set(), 'cs': set()
    })

    for row in results_attr:
        m = str(row.mof)
        if row.formula: attr_collector[m]['formula'].add(str(row.formula))
        if row.mofid: attr_collector[m]['mofid'].add(str(row.mofid))
        if row.topo: attr_collector[m]['topo'].add(str(row.topo))
        if row.metal: attr_collector[m]['metal'].add(str(row.metal))
        if row.smiles: attr_collector[m]['smiles'].add(str(row.smiles))
        if row.sg: attr_collector[m]['sg'].add(str(row.sg))
        if row.cs: attr_collector[m]['cs'].add(str(row.cs))

    # Convert sets to joined strings
    processed_attrs = []
    for m, attrs in attr_collector.items():
        processed_attrs.append({
            'mof_uri': m,
            'chemical_formula': "; ".join(sorted(attrs['formula'])),
            'mofid': "; ".join(sorted(attrs['mofid'])),
            'topology': "; ".join(sorted(attrs['topo'])),
            'metal_cluster_elements': "; ".join(sorted(attrs['metal'])),
            'linker_smiles': "; ".join(sorted(attrs['smiles'])),
            'space_group': "; ".join(sorted(attrs['sg'])),
            'crystal_system': "; ".join(sorted(attrs['cs']))
        })

    # 6. Pivot and Merge
    print("Finalizing dataframe...")
    df_num = pd.DataFrame(numeric_data)
    df_attr = pd.DataFrame(processed_attrs)
    
    if not df_num.empty:
        pivot_df = df_num.pivot_table(
            index='mof_uri', 
            columns='property_name', 
            values='property_value',
            aggfunc='first'
        ).reset_index()
    else:
        pivot_df = pd.DataFrame(columns=['mof_uri'])

    # Merge attributes with numerical properties
    final_df = df_attr.merge(pivot_df, on='mof_uri', how='outer')
    
    # Add CSD codes if available
    csd_map = {}
    if csd_map:
        final_df['csd_code'] = final_df['mof_uri'].map(csd_map)

    # 7. Save to CSV
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved all properties to {OUTPUT_PATH}")
    print(f"Final output shape: {final_df.shape}")
    print(f"Total unique MOFs: {len(final_df)}")

if __name__ == "__main__":
    main()












