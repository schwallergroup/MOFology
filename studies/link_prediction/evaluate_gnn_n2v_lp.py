import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from rdflib import Graph, URIRef
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# CONFIG
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
# Paths to the embeddings we want to evaluate
EMB_PATHS = {
    "CompGCN": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_compgcn_embeddings_256d_2layers.csv),
    "Node2Vec": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_p1.0_q1.0.csv),
    "TransE": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings"/mof_transe_embeddings_256d.csv)
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "link_prediction_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INVERSE_RELATIONS_TO_DROP = {
    "http://emmo.info/domain-mof/mof-ontology#usedInMOF",
    "http://emmo.info/domain-mof/mof-ontology#isComponentOf",
    "http://emmo.info/domain-mof/mof-ontology#hasPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasComputationalPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasStructuralPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#hasPhysicalPropertyOwner",
    "http://emmo.info/domain-mof/mof-ontology#describedIn",
    "http://emmo.info/domain-mof/mof-ontology#describedInAbstract",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range"
}

def get_hetero_graph_from_kg():
    logging.info(f"Loading KG from {KG_PATH}...")
    g = Graph()
    g.parse(KG_PATH, format="turtle")

    nodes = set()
    relations = set()
    triplets = [] 

    logging.info("Parsing triples and filtering inverses...")
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            p_str = str(p)
            if p_str in INVERSE_RELATIONS_TO_DROP or "Owner" in p_str or "usedIn" in p_str or "isComponent" in p_str:
                continue
            s_str, o_str = str(s), str(o)
            nodes.add(s_str)
            nodes.add(o_str)
            relations.add(p_str)
            triplets.append((s_str, p_str, o_str))

    node_to_idx = {node: i for i, node in enumerate(sorted(nodes))}
    rel_to_idx = {rel: i for i, rel in enumerate(sorted(relations))}
    
    src, dst, edge_types = [], [], []
    for s, p, o in triplets:
        src.append(node_to_idx[s])
        dst.append(node_to_idx[o])
        edge_types.append(rel_to_idx[p])
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    logging.info(f"Final Graph: {len(nodes)} nodes, {len(relations)} relations, {len(triplets)} edges.")
    return edge_index, edge_type, node_to_idx, rel_to_idx

def load_embeddings(node_to_idx, csv_path):
    logging.info(f"Loading embeddings from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Handle different column names for URI
    if 'mof_uri' in df.columns:
        df = df.set_index('mof_uri')
    elif 'uri' in df.columns:
        df = df.set_index('uri')
    else:
        df = df.set_index(df.columns[0])
    
    # Identify embedding columns (numeric)
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        # Fallback for when read_csv fails to identify numeric columns due to mixed types
        # This happens if metadata columns have 'set()' or other strings
        emb_cols = [c for c in df.columns if c.startswith('emb_') or c.startswith('rel_emb_')]
        df_numeric = df[emb_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    x = torch.zeros((len(node_to_idx), df_numeric.shape[1]), dtype=torch.float32)
    match_count = 0
    for uri, idx in node_to_idx.items():
        if uri in df_numeric.index:
            vals = df_numeric.loc[uri].values
            if vals.ndim > 1: vals = vals[0] # Handle duplicates
            x[idx] = torch.tensor(vals.astype(np.float32))
            match_count += 1
            
    logging.info(f"Loaded {match_count}/{len(node_to_idx)} embeddings. Dim: {x.size(1)}")
    return x

def evaluate_downstream(name, x, edge_index, node_to_idx):
    logging.info(f"Starting downstream evaluation for {name}...")
    
    num_samples = min(50000, edge_index.size(1))
    indices = torch.randperm(edge_index.size(1))[:num_samples]
    
    pos_h = edge_index[0, indices]
    pos_t = edge_index[1, indices]
    
    neg_t = torch.randint(0, len(node_to_idx), (num_samples,))
    
    def get_feats(h, t):
        h_vec = x[h]
        t_vec = x[t]
        if name == "TransE":
            # TransE is translation-based, so (h - t) is a powerful feature
            diff_vec = h_vec - t_vec
            return torch.cat([h_vec, t_vec, diff_vec], dim=1).numpy()
        else:
            # For Node2Vec and GNN, we use concat(h, t)
            return torch.cat([h_vec, t_vec], dim=1).numpy()

    X_pos = get_feats(pos_h, pos_t)
    X_neg = get_feats(pos_h, neg_t)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*num_samples + [0]*num_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "XGBoost": XGBClassifier(n_jobs=-1),
        "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500)
    }
    
    results = []
    for m_name, model in models.items():
        logging.info(f"  Training {m_name}...")
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        logging.info(f"  {m_name} AUC: {auc:.4f}")
        results.append({"Model": m_name, "AUC": auc, "Embedding": name})
        
    return results

def main():
    edge_index, edge_type, node_to_idx, rel_to_idx = get_hetero_graph_from_kg()
    
    all_results = []
    for name, path in EMB_PATHS.items():
        if not os.path.exists(path):
            logging.warning(f"Embedding file not found: {path}")
            continue
        x = load_embeddings(node_to_idx, path)
        results = evaluate_downstream(name, x, edge_index, node_to_idx)
        all_results.extend(results)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUTPUT_DIR}/link_prediction_comparison.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="Model", y="AUC", hue="Embedding")
    plt.title("Downstream Link Prediction Performance Comparison")
    plt.ylim(0.5, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lp_comparison_plot.png")
    logging.info(f"Study complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
