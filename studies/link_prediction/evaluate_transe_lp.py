import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from rdflib import Graph, URIRef
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression # Ridge equivalent for classification

# -----------------------------
# CONFIG
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
TRANSE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "transe_embeddings"/mof_transe_embeddings_256d.csv)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "link_prediction_transe")

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_transe_features(node_to_idx, csv_path):
    logging.info(f"Loading TransE embeddings from {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.set_index('mof_uri')
    
    # TransE export uses emb_0, emb_1... columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    x = torch.zeros((len(node_to_idx), df_numeric.shape[1]), dtype=torch.float32)
    match_count = 0
    for uri, idx in node_to_idx.items():
        if uri in df.index:
            x[idx] = torch.tensor(df.loc[uri].values.astype(np.float32))
            match_count += 1
            
    logging.info(f"Loaded {match_count}/{len(node_to_idx)} embeddings. Dim: {x.size(1)}")
    return x

# -----------------------------
# DOWNSTREAM CLASS EVALUATION
# -----------------------------
def evaluate_downstream(x, edge_index, edge_type, node_to_idx, rel_to_idx):
    logging.info("Starting downstream Link Prediction comparison...")
    
    # Prepare data for classification: (h, r, t) -> 1, (h, r, random_t) -> 0
    num_samples = min(50000, edge_index.size(1))
    indices = torch.randperm(edge_index.size(1))[:num_samples]
    
    pos_h = edge_index[0, indices]
    pos_t = edge_index[1, indices]
    pos_r = edge_type[indices]
    
    neg_t = torch.randint(0, len(node_to_idx), (num_samples,))
    
    # Features: concat(h_emb, t_emb, h_emb - t_emb)
    # TransE is translation-based (h + r = t), so (h - t) is a powerful feature
    # that represents the relationship vector 'r'.
    def get_feats(h, t):
        h_vec = x[h]
        t_vec = x[t]
        diff_vec = h_vec - t_vec
        return torch.cat([h_vec, t_vec, diff_vec], dim=1).numpy()

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
    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        logging.info(f"{name} AUC: {auc:.4f}")
        results.append({"Model": name, "AUC": auc})
        
    return pd.DataFrame(results)

def main():
    edge_index, edge_type, node_to_idx, rel_to_idx = get_hetero_graph_from_kg()
    x_transe = load_transe_features(node_to_idx, TRANSE_PATH)
    
    results_df = evaluate_downstream(x_transe, edge_index, edge_type, node_to_idx, rel_to_idx)
    results_df.to_csv(f"{OUTPUT_DIR}/transe_link_prediction_comparison.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Model", y="AUC")
    plt.title("TransE Link Prediction Downstream Performance")
    plt.ylim(0.5, 1.0)
    plt.savefig(f"{OUTPUT_DIR}/transe_lp_comparison.png")
    logging.info(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
