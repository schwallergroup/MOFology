import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import ast  # ADDED: To parse stringified lists
from rdflib import Graph, URIRef
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import RGCNConv
from sklearn.metrics import roc_auc_score

# -----------------------------
# CONFIG
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
N2V_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_p1.0_q1.0.csv)
GPS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_advanced_embeddings_256d_4layers_1000epochs.csv)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "link_prediction_hetero_filtered")

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# DATA LEAKAGE PREVENTION
# -----------------------------
INVERSE_RELATIONS_TO_DROP = {
    "http://emmo.info/domain-mof/mof-ontology#usedInMOF",
    "http://emmo.info/domain-mof/mof-ontology#isComponentOf",
    "http://emmo.info/domain-mof/mof-ontology#hasPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasComputationalPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasStructuralPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasPhysicalPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#describedIn",
    "http://emmo.info/domain-mof/mof-ontology#describedInAbstract",
    "http://emmo.info/domain-mof/mof-ontology#usedInSynthesis",
    "http://emmo.info/domain-mof/mof-ontology#usedAsMetalPrecursorIn",
    "http://emmo.info/domain-mof/mof-ontology#usedAsLinkerPrecursorIn",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf"
}

# -----------------------------
# DATA LOADING
# -----------------------------
def get_hetero_graph_from_kg():
    logging.info(f"Loading KG from {KG_PATH}...")
    g = Graph()
    try:
        g.parse(KG_PATH, format="turtle")
    except Exception as e:
        logging.error(f"Failed to load KG: {e}")
        return None, None, None, None

    nodes = set()
    relations = set()
    triplets = [] 

    logging.info("Parsing triples and filtering inverses...")
    skipped_count = 0
    
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            p_str = str(p)
            if p_str in INVERSE_RELATIONS_TO_DROP:
                skipped_count += 1
                continue
            if "Owner" in p_str or "usedIn" in p_str or "isComponent" in p_str:
                skipped_count += 1
                continue

            s_str, o_str = str(s), str(o)
            nodes.add(s_str)
            nodes.add(o_str)
            relations.add(p_str)
            triplets.append((s_str, p_str, o_str))

    logging.info(f"Dropped {skipped_count} inverse/redundant edges to prevent leakage.")
    
    node_to_idx = {node: i for i, node in enumerate(sorted(nodes))}
    rel_to_idx = {rel: i for i, rel in enumerate(sorted(relations))}
    
    src = []
    dst = []
    edge_types = []
    
    for s, p, o in triplets:
        src.append(node_to_idx[s])
        dst.append(node_to_idx[o])
        edge_types.append(rel_to_idx[p])
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    logging.info(f"Final Graph: {len(nodes)} nodes, {len(relations)} relations, {len(triplets)} edges.")
    return edge_index, edge_type, node_to_idx, rel_to_idx

def load_node_features(node_to_idx, csv_path):
    if not os.path.exists(csv_path): 
        logging.warning(f"Path not found: {csv_path}")
        return None
    
    logging.info(f"Loading embeddings from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        
        # Index Logic
        if 'mof_uri' in df.columns:
            df = df.set_index('mof_uri')
        elif 'uri' in df.columns:
            df = df.set_index('uri')
        elif df.iloc[:,0].dtype == object: 
            df = df.set_index(df.columns[0])
            
        # Check if we have a "stringified list" column (common in some exports)
        # e.g. column "embedding" with values "[0.1, 0.2, ...]"
        string_col = None
        for col in df.columns:
            if df[col].dtype == object and isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):
                string_col = col
                break
        
        x_list = []
        valid_indices = []
        
        if string_col:
            logging.info(f"Found stringified embedding column: {string_col}. Parsing...")
            # Parse row by row
            for uri, idx in node_to_idx.items():
                if uri in df.index:
                    val_str = df.loc[uri, string_col]
                    if isinstance(val_str, pd.Series): val_str = val_str.iloc[0] # Handle dupes
                    try:
                        # Fast/Safe parsing of literal list
                        vec = np.array(ast.literal_eval(val_str), dtype=np.float32)
                        x_list.append((idx, vec))
                    except:
                        pass
        else:
            # Standard numeric columns
            df_numeric = df.select_dtypes(include=[np.number])
            if df_numeric.empty:
                # Fallback: try parsing all cols as numeric
                df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

            if df_numeric.empty:
                logging.error("No numeric data found.")
                return None
                
            logging.info(f"Using {df_numeric.shape[1]} numeric columns as features.")
            
            for uri, idx in node_to_idx.items():
                if uri in df_numeric.index:
                    vals = df_numeric.loc[uri].values
                    if vals.ndim > 1: vals = vals[0]
                    x_list.append((idx, vals.astype(np.float32)))

        if not x_list:
            logging.error("No embeddings matched with graph nodes.")
            return None

        # Create Tensor
        emb_dim = len(x_list[0][1])
        x = torch.zeros((len(node_to_idx), emb_dim), dtype=torch.float32)
        
        for idx, vec in x_list:
            if len(vec) == emb_dim:
                x[idx] = torch.tensor(vec)
                
        logging.info(f"Loaded {len(x_list)}/{len(node_to_idx)} embeddings. Dim: {emb_dim}")
        return x
        
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None

# -----------------------------
# MODEL
# -----------------------------
class RGCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, use_embeddings=False, num_nodes=None):
        super().__init__()
        self.use_embeddings = use_embeddings
        if self.use_embeddings:
            self.emb = torch.nn.Embedding(num_nodes, in_channels)
            
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.rel_embedding = torch.nn.Embedding(num_relations, hidden_channels)

    def encode(self, x, edge_index, edge_type):
        if self.use_embeddings:
            x = self.emb(x)
        x = self.conv1(x, edge_index, edge_type).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x

    def decode(self, z, edge_index, edge_type):
        source = z[edge_index[0]]
        target = z[edge_index[1]]
        rel = self.rel_embedding(edge_type)
        return (source * target * rel).sum(dim=-1)

# -----------------------------
# MAIN LOOP
# -----------------------------
def run_experiment(name, x, edge_index, edge_type, num_nodes, num_rels, idx_to_rel):
    logging.info(f"\n=== Heterogeneous Experiment: {name} ===")
    
    if x is not None and x.size(1) == 0:
        logging.error(f"Skipping {name}: Feature dim is 0.")
        return pd.DataFrame()

    data = Data(
        x=torch.arange(num_nodes) if x is None else x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes
    )
    
    indices = torch.randperm(data.num_edges)
    test_size = int(0.1 * data.num_edges)
    val_size = int(0.1 * data.num_edges)
    train_size = data.num_edges - test_size - val_size
    
    train_mask = indices[:train_size]
    test_mask = indices[train_size+val_size:]
    
    train_data = Data(edge_index=edge_index[:, train_mask], edge_type=edge_type[train_mask], num_nodes=num_nodes)
    
    in_dim = 64 if x is None else x.size(1)
    
    model = RGCNLinkPredictor(
        in_channels=in_dim,
        hidden_channels=64,
        num_relations=num_rels,
        use_embeddings=(x is None),
        num_nodes=num_nodes
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        x_in = data.x.to(DEVICE)
        
        z = model.encode(x_in, train_data.edge_index.to(DEVICE), train_data.edge_type.to(DEVICE))
        pos_out = model.decode(z, train_data.edge_index.to(DEVICE), train_data.edge_type.to(DEVICE))
        
        neg_tails = torch.randint(0, num_nodes, (train_data.edge_index.size(1),), device=DEVICE)
        neg_index = torch.stack([train_data.edge_index[0].to(DEVICE), neg_tails])
        neg_out = model.decode(z, neg_index, train_data.edge_type.to(DEVICE))
        
        out = torch.cat([pos_out, neg_out])
        y = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        loss = F.binary_cross_entropy_with_logits(out, y)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}: Loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(x_in, data.edge_index.to(DEVICE), data.edge_type.to(DEVICE))
        
        test_edges = edge_index[:, test_mask].to(DEVICE)
        test_types = edge_type[test_mask].to(DEVICE)
        
        neg_tails = torch.randint(0, num_nodes, (test_edges.size(1),), device=DEVICE)
        test_neg_edges = torch.stack([test_edges[0], neg_tails])
        
        pos_scores = model.decode(z, test_edges, test_types)
        neg_scores = model.decode(z, test_neg_edges, test_types)
        
        y_true = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).cpu()
        y_pred = torch.cat([pos_scores, neg_scores]).sigmoid().cpu()
        global_auc = roc_auc_score(y_true, y_pred)
        logging.info(f"Global Test AUC: {global_auc:.4f}")
        
        rel_metrics = []
        unique_test_types = torch.unique(test_types).cpu().numpy()
        
        for rid in unique_test_types:
            mask = (test_types.cpu() == rid)
            if mask.sum() < 5: continue
            
            p = pos_scores.cpu()[mask]
            n = neg_scores.cpu()[mask]
            
            y_r = torch.cat([torch.ones_like(p), torch.zeros_like(n)])
            y_p = torch.cat([p, n])
            
            try:
                auc = roc_auc_score(y_r, y_p)
                r_name = idx_to_rel[rid].split('#')[-1]
                rel_metrics.append({'Relation': r_name, 'AUC': auc, 'Count': mask.sum().item()})
            except:
                pass
                
        df = pd.DataFrame(rel_metrics)
        logging.info("\nPer-Relation Accuracy:")
        logging.info(df)
        df['Model'] = name
        return df

def main():
    edge_index, edge_type, node_to_idx, rel_to_idx = get_hetero_graph_from_kg()
    if edge_index is None: return
    
    idx_to_rel = {v: k for k, v in rel_to_idx.items()}
    num_nodes = len(node_to_idx)
    num_rels = len(rel_to_idx)
    
    metrics = []

    # 1. Scratch
    res = run_experiment("Scratch", None, edge_index, edge_type, num_nodes, num_rels, idx_to_rel)
    metrics.append(res)
    
    # 2. Node2Vec
    x_n2v = load_node_features(node_to_idx, N2V_PATH)
    if x_n2v is not None:
        res = run_experiment("Node2Vec", x_n2v, edge_index, edge_type, num_nodes, num_rels, idx_to_rel)
        metrics.append(res)

    # 3. GraphGPS
    x_gps = load_node_features(node_to_idx, GPS_PATH)
    if x_gps is not None:
        res = run_experiment("GraphGPS", x_gps, edge_index, edge_type, num_nodes, num_rels, idx_to_rel)
        metrics.append(res)
    
    final_df = pd.concat(metrics)
    final_df.to_csv(f"{OUTPUT_DIR}/relation_specific_metrics.csv", index=False)
    logging.info(f"Final results saved to {OUTPUT_DIR}/relation_specific_metrics.csv")

if __name__ == "__main__":
    main()