import os
import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import logging
import gc
import rdflib
from rdflib import Graph
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths - TTL file path (can be passed as argument or set here)
TTL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "node2vec/Mar2026")

# Parameter sweep configurations: (p, q)
PARAM_SWEEP = [
    (1.0, 0.5),
    (0.5, 2.0),
    (1.0, 1.0),
    (2.0, 0.5)
]

def load_ttl_to_networkx(ttl_path):
    """
    Load TTL file and convert to NetworkX graph.
    
    Args:
        ttl_path: Path to TTL file
    
    Returns:
        NetworkX graph
    """
    logging.info(f"Loading TTL file from {ttl_path}...")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    logging.info(f"Loaded {len(g)} triples from TTL file.")
    
    logging.info("Converting RDF graph to NetworkX graph...")
    nx_graph = nx.Graph()
    for s, p, o in tqdm(g, desc="Building NetworkX graph"):
        # Only add edges where both subject and object are nodes (not literals)
        if isinstance(o, (rdflib.URIRef, rdflib.BNode)):
            nx_graph.add_edge(str(s), str(o))
    
    del g
    gc.collect()
    
    logging.info(f"NetworkX graph created: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges.")
    return nx_graph

def extract_mof_metadata(ttl_path):
    """
    Extract MOF metadata from TTL file.
    
    Args:
        ttl_path: Path to TTL file
    
    Returns:
        List of dictionaries with MOF metadata
    """
    logging.info("Extracting MOF metadata from TTL file...")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    
    query = """
    PREFIX : <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?csd ?prop ?linker_smiles ?metal
    WHERE {
        ?mof a ?type .
        FILTER(?type IN (:MOF, :ExperimentalMOF, :HypotheticalMOF))
        OPTIONAL { ?mof :hasCSDCode ?csd }
        OPTIONAL { 
            ?mof :hasStructuralProperty ?prop .
            FILTER(CONTAINS(STR(?prop), "SpaceGroup")) 
        }
        OPTIONAL { 
            ?mof :hasLinker ?linkerObj .
            ?linkerObj :hasSMILES ?linker_smiles .
        }
        OPTIONAL { 
            ?mof :hasMetalNode ?metalObj .
            BIND(STR(?metalObj) AS ?metal)
        }
    }
    """
    
    results = list(g.query(query))
    mof_dict = {}
    
    for row in tqdm(results, desc="Processing MOF metadata"):
        mof_uri = str(row.mof)
        if mof_uri not in mof_dict:
            mof_dict[mof_uri] = {
                "mof_uri": mof_uri,
                "csd_code": str(row.csd) if row.csd else None,
                "space_groups": set(),
                "linkers": set(),
                "metals": set()
            }
        if row.prop:
            sg = str(row.prop).split("#")[-1]
            mof_dict[mof_uri]["space_groups"].add(sg)
        if row.linker_smiles:
            mof_dict[mof_uri]["linkers"].add(str(row.linker_smiles))
        if row.metal:
            metal = str(row.metal).split("#")[-1]
            mof_dict[mof_uri]["metals"].add(metal)
    
    mof_data = []
    for mof_uri, data in mof_dict.items():
        data["space_group"] = "; ".join(sorted(data["space_groups"])) if data["space_groups"] else "N/A"
        data["linkers"] = "; ".join(sorted(data["linkers"])) if data["linkers"] else "N/A"
        data["metal_node"] = "; ".join(sorted(data["metals"])) if data["metals"] else "N/A"
        mof_data.append(data)
    
    logging.info(f"Found {len(mof_data)} MOFs in TTL file.")
    return mof_data

def generate_embeddings(nx_graph, p=1.0, q=1.0):
    """
    Generate node2vec embeddings for the graph with specified p and q parameters.
    
    Args:
        nx_graph: NetworkX graph
        p: Return parameter (controls likelihood of returning to previous node)
        q: In-out parameter (controls likelihood of exploring new nodes)
    
    Returns:
        Trained node2vec model
    """
    logging.info(f"Initializing Node2Vec with p={p}, q={q}...")
    if nx_graph.number_of_edges() == 0:
        logging.warning("Graph has no edges!")
        return None

    node2vec = Node2Vec(
        nx_graph, 
        dimensions=32, 
        walk_length=5, 
        num_walks=5, 
        workers=32, 
        p=p, 
        q=q, 
        quiet=False
    )
    
    logging.info("Fitting model...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4, workers=32)
    return model

def extract_mof_embeddings(model, mof_data):
    """
    Extract embeddings for MOF nodes only.
    
    Args:
        model: Trained node2vec model
        mof_data: List of MOF metadata dictionaries
    
    Returns:
        DataFrame with MOF metadata and embeddings
    """
    logging.info("Extracting MOF embeddings...")
    embeddings_list = []
    
    for mof in tqdm(mof_data, desc="Extracting embeddings"):
        mof_uri = mof["mof_uri"]
        if str(mof_uri) in model.wv:
            embedding = model.wv[str(mof_uri)]
            row = mof.copy()
            for i, val in enumerate(embedding):
                row[f"emb_{i}"] = val
            embeddings_list.append(row)
        else:
            logging.warning(f"MOF {mof_uri} not found in model vocabulary")
    
    df = pd.DataFrame(embeddings_list)
    return df

def main(ttl_path=None):
    if ttl_path is None:
        ttl_path = TTL_PATH
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load TTL file and convert to NetworkX
    nx_graph = load_ttl_to_networkx(ttl_path)
    
    # Extract MOF metadata
    mof_data = extract_mof_metadata(ttl_path)
    
    # Run parameter sweep
    for p, q in PARAM_SWEEP:
        logging.info(f"\n{'='*60}")
        logging.info(f"Running parameter sweep: p={p}, q={q}")
        logging.info(f"{'='*60}")
        
        # Generate embeddings with current parameters
        model = generate_embeddings(nx_graph, p=p, q=q)
        
        if not model:
            logging.error(f"Failed to generate model for p={p}, q={q}")
            continue
        
        # Extract MOF embeddings only
        df_embeddings = extract_mof_embeddings(model, mof_data)
        
        # Save with p and q in filename
        output_file = os.path.join(OUTPUT_DIR, f"mof_embeddings_p{p}_q{q}.csv")
        logging.info(f"Saving node2vec embeddings for {len(df_embeddings)} MOFs to {output_file}...")
        df_embeddings.to_csv(output_file, index=False)
        
        # Clean up model to free memory
        del model
        gc.collect()
        logging.info(f"Completed parameter sweep for p={p}, q={q}")
    
    logging.info("\n" + "="*60)
    logging.info("All parameter sweeps completed!")
    logging.info("="*60)

if __name__ == "__main__":
    import sys
    ttl_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(ttl_path)