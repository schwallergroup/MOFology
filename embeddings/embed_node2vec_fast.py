import os
import pandas as pd
import logging
import gc
import rdflib
from rdflib import Graph
from pecanpy import pecanpy
from gensim.models import Word2Vec
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
TTL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec")
TEMP_EDGELIST = os.path.join(OUTPUT_DIR, "temp_graph.edgelist")  # Use absolute path

# Performance Params
DIMENSIONS = 32
WALK_LENGTH = 10
NUM_WALKS = 5
WORKERS = 32

PARAM_SWEEP = [(1.0, 0.5), (0.5, 2.0), (1.0, 1.0), (2.0, 0.5)]

def extract_mof_metadata(ttl_path):
    """Extract MOF metadata using proper RDF parsing."""
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

def build_edgelist_file(ttl_path, out_edgelist):
    """Parse TTL using rdflib and write integer-based edgelist with TAB delimiter."""
    logging.info("Loading TTL file with rdflib...")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    logging.info(f"Loaded {len(g)} triples.")
    
    logging.info("Extracting edges and mapping IDs...")
    node_to_id = {}
    id_to_node = {}
    current_id = 0
    edge_count = 0

    def get_id(node_str):
        nonlocal current_id
        if node_str not in node_to_id:
            node_to_id[node_str] = current_id
            id_to_node[current_id] = node_str
            current_id += 1
        return node_to_id[node_str]

    with open(out_edgelist, 'w') as f_out:
        for s, p, o in tqdm(g, desc="Building edgelist"):
            # Only add edges where both subject and object are nodes (not literals)
            if isinstance(o, (rdflib.URIRef, rdflib.BNode)):
                s_str = str(s)
                o_str = str(o)
                # Skip self-loops
                if s_str == o_str:
                    continue
                s_id = get_id(s_str)
                o_id = get_id(o_str)
                # CRITICAL: Use TAB delimiter (pecanpy expects tabs by default)
                f_out.write(f"{s_id}\t{o_id}\n")
                edge_count += 1
                
    logging.info(f"Edgelist created with {edge_count} edges and {len(node_to_id)} nodes.")
    
    # Validate the edgelist file
    logging.info("Validating edgelist file...")
    invalid_lines = []
    with open(out_edgelist, 'r') as f_in:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                invalid_lines.append((line_num, "empty line"))
                continue
            parts = line.split('\t')  # Check tab delimiter
            if len(parts) != 2:
                # Try space as fallback check
                parts = line.split()
                if len(parts) != 2:
                    invalid_lines.append((line_num, f"expected 2 values, got {len(parts)}: '{line}'"))
            else:
                try:
                    int(parts[0])
                    int(parts[1])
                except ValueError:
                    invalid_lines.append((line_num, f"non-integer IDs: '{line}'"))
    
    if invalid_lines:
        logging.error(f"Found {len(invalid_lines)} invalid lines in edgelist:")
        for line_num, error in invalid_lines[:10]:  # Show first 10 errors
            logging.error(f"  Line {line_num}: {error}")
        if len(invalid_lines) > 10:
            logging.error(f"  ... and {len(invalid_lines) - 10} more")
        raise ValueError("Edgelist validation failed")
    
    logging.info("Edgelist validation passed.")
    return node_to_id, id_to_node

def clean_edgelist(edgelist_path):
    """Clean edgelist file to ensure tab delimiter and no empty lines."""
    logging.info("Cleaning edgelist file...")
    temp_path = edgelist_path + ".tmp"
    
    valid_lines = 0
    with open(edgelist_path, 'r') as f_in, open(temp_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split('\t')  # Try tab first
            if len(parts) != 2:
                parts = line.split()  # Fallback to any whitespace
                if len(parts) != 2:
                    continue  # Skip invalid lines
            try:
                # Validate integers
                int(parts[0])
                int(parts[1])
                # Write with tab delimiter
                f_out.write(f"{parts[0]}\t{parts[1]}\n")
                valid_lines += 1
            except ValueError:
                continue  # Skip lines with non-integer values
    
    os.replace(temp_path, edgelist_path)
    logging.info(f"Cleaned edgelist: {valid_lines} valid edges")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Extract MOF metadata first (using proper RDF parsing)
    mof_data = extract_mof_metadata(TTL_PATH)
    mof_uris = {mof["mof_uri"] for mof in mof_data}
    
    # 2. Build edgelist (using proper RDF parsing)
    node_to_id, id_to_node = build_edgelist_file(TTL_PATH, TEMP_EDGELIST)
    
    # 3. Clean edgelist to ensure tab delimiter (safety check)
    clean_edgelist(TEMP_EDGELIST)
    
    # 4. Create reverse mapping: URI -> ID string (for Word2Vec lookup)
    uri_to_id_str = {uri: str(node_id) for node_id, uri in id_to_node.items()}
    
    # 5. Run parameter sweep
    for p, q in PARAM_SWEEP:
        logging.info(f"\n{'='*60}")
        logging.info(f"Training: p={p}, q={q}")
        logging.info(f"{'='*60}")
        
        try:
            # Initialize pecanpy graph
            logging.info("Initializing pecanpy SparseOTF graph...")
            g = pecanpy.SparseOTF(p=p, q=q, workers=WORKERS, extend=True)
            
            logging.info(f"Loading graph from {TEMP_EDGELIST}...")
            # CRITICAL: Use tab delimiter explicitly
            g.read_edg(TEMP_EDGELIST, weighted=False, directed=False, delimiter='\t')
            logging.info(f"Graph loaded successfully: {g.num_nodes} nodes, {g.num_edges} edges")
            
            logging.info("Generating random walks...")
            walks = g.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LENGTH)
            logging.info(f"Generated {len(walks)} walks")
            
            logging.info("Training Word2Vec model...")
            model = Word2Vec(
                sentences=walks, 
                vector_size=DIMENSIONS, 
                window=10, 
                min_count=1, 
                sg=1, 
                workers=WORKERS
            )
            logging.info(f"Model trained. Vocabulary size: {len(model.wv)}")
            
            # 6. Extract MOF embeddings and combine with metadata
            logging.info("Extracting MOF embeddings...")
            embeddings_list = []
            missing_count = 0
            
            for mof in tqdm(mof_data, desc="Extracting embeddings"):
                mof_uri = mof["mof_uri"]
                mof_id_str = uri_to_id_str.get(mof_uri)
                
                if mof_id_str and mof_id_str in model.wv:
                    embedding = model.wv[mof_id_str]
                    row = mof.copy()
                    for i, val in enumerate(embedding):
                        row[f"emb_{i}"] = val
                    embeddings_list.append(row)
                else:
                    missing_count += 1
                    if missing_count <= 5:  # Only log first few
                        logging.warning(f"MOF {mof_uri} not found in model vocabulary")
            
            if missing_count > 0:
                logging.warning(f"{missing_count} MOFs were not found in the model vocabulary")
            
            # 7. Save with metadata
            df = pd.DataFrame(embeddings_list)
            out_path = os.path.join(OUTPUT_DIR, f"mof_embeddings_p{p}_q{q}.csv")
            df.to_csv(out_path, index=False)
            
            logging.info(f"✓ Saved {len(df)} MOF embeddings to {out_path}")
            
        except Exception as e:
            logging.error(f"Error during processing for p={p}, q={q}: {e}", exc_info=True)
            raise
        
        finally:
            # Clean up
            if 'g' in locals():
                del g
            if 'walks' in locals():
                del walks
            if 'model' in locals():
                del model
            if 'df' in locals():
                del df
            gc.collect()

    # Cleanup temporary file
    if os.path.exists(TEMP_EDGELIST):
        os.remove(TEMP_EDGELIST)
        logging.info("Cleaned up temporary edgelist file")
    
    logging.info("\n" + "="*60)
    logging.info("All parameter sweeps completed!")
    logging.info("="*60)

if __name__ == "__main__":
    main()