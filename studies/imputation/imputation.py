import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from rdflib import Graph, URIRef, Namespace

# -----------------------------
# CONFIG
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
N2V_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_p1.0_q1.0.csv)
GPS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_advanced_embeddings_256d_4layers_1000epochs.csv)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "imputation_structural")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. EXTRACT LABELS FROM KG
# -----------------------------
def get_structural_labels_from_kg():
    """Parses the KG to find structural properties for MOFs."""
    logging.info(f"Loading KG from {KG_PATH} to extract labels...")
    g = Graph()
    try:
        g.parse(KG_PATH, format="turtle")
    except Exception as e:
        logging.error(f"Failed to load KG: {e}")
        return pd.DataFrame()

    MOF = Namespace("http://emmo.info/domain-mof/mof-ontology#")
    
    # Store labels
    data = []
    
    # Query for Topology and Metal Nodes
    # We iterate manually or use SPARQL. SPARQL is cleaner here.
    query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    SELECT ?mof ?topology ?metal WHERE {
        ?mof a mof:MOF .
        OPTIONAL { 
            ?mof mof:hasTopology ?topoNode . 
            ?topoNode mof:topologyCode ?topology .
        }
        OPTIONAL {
            ?mof mof:hasMetalNode ?node .
            ?node mof:hasMetalElement ?metal .
        }
    }
    """
    
    logging.info("Querying KG for structural properties...")
    for row in g.query(query):
        data.append({
            'mof_uri': str(row.mof),
            'topology': str(row.topology) if row.topology else None,
            'metal_element': str(row.metal) if row.metal else None
        })
        
    df = pd.DataFrame(data)
    
    # Drop rows where everything is null (except URI)
    df = df.dropna(subset=['topology', 'metal_element'], how='all')
    
    # Group by MOF to handle multiple metals (e.g. bimetallic)
    # For simplicity in classification, we'll just take the primary (first) metal if multiple exist,
    # or create a combined label string.
    df = df.groupby('mof_uri').agg({
        'topology': 'first', 
        'metal_element': lambda x: sorted(list(set(x.dropna())))[0] if len(x.dropna()) > 0 else None
    }).reset_index()
    
    logging.info(f"Extracted labels for {len(df)} MOFs.")
    return df.set_index('mof_uri')

# -----------------------------
# 2. LOAD EMBEDDINGS (ROBUST)
# -----------------------------
def load_embeddings(csv_path):
    """Loads embeddings with robust parsing for stringified lists."""
    if not os.path.exists(csv_path):
        logging.warning(f"Embedding file not found: {csv_path}")
        return None
        
    logging.info(f"Loading embeddings from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        
        # Index Logic
        if 'mof_uri' in df.columns:
            df = df.set_index('mof_uri')
        elif 'uri' in df.columns:
            df = df.set_index('uri')
        elif df.iloc[:,0].dtype == object: 
            df = df.set_index(df.columns[0])
            
        # Check for stringified list column
        string_col = None
        for col in df.columns:
            if df[col].dtype == object and isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):
                string_col = col
                break
        
        if string_col:
            logging.info(f"Parsing stringified column: {string_col}")
            # Expand list column into multiple feature columns
            # This can be slow, so we do it carefully
            features = []
            indices = []
            for idx, row in df.iterrows():
                try:
                    vec = ast.literal_eval(row[string_col])
                    features.append(vec)
                    indices.append(idx)
                except:
                    pass
            return pd.DataFrame(features, index=indices)
        else:
            # Standard numeric columns
            df_numeric = df.select_dtypes(include=[np.number])
            if df_numeric.empty:
                # Fallback
                df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            return df_numeric

    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None

# -----------------------------
# 3. IMPUTATION TASK
# -----------------------------
def run_imputation(target_name, df_labels, df_emb, emb_name, min_samples=10):
    logging.info(f"--- Predicting {target_name} using {emb_name} ---")
    
    # 1. Align Data
    # Get target column
    if target_name not in df_labels.columns:
        logging.warning(f"Target {target_name} not in labels.")
        return None
        
    y = df_labels[target_name].dropna()
    
    # Filter rare classes
    counts = y.value_counts()
    valid_classes = counts[counts >= min_samples].index
    y = y[y.isin(valid_classes)]
    
    if len(y) == 0:
        logging.warning("No valid samples after filtering.")
        return None
        
    # Intersect with embeddings
    common = y.index.intersection(df_emb.index)
    y = y.loc[common]
    X = df_emb.loc[common]
    
    if len(y) < 50:
        logging.warning(f"Insufficient samples ({len(y)}) for training.")
        return None
        
    logging.info(f"Samples: {len(y)}, Classes: {len(valid_classes)}")
    
    # 2. Split
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # 3. Train Classifier
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = clf.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    
    # Baseline (Most Frequent)
    dummy_pred = np.full_like(y_test, Counter(y_train).most_common(1)[0][0])
    base_acc = accuracy_score(y_test, dummy_pred)
    
    logging.info(f"Accuracy: {acc:.4f} (Baseline: {base_acc:.4f}) | F1: {f1:.4f}")
    
    return {
        'Target': target_name,
        'Embedding': emb_name,
        'Accuracy': acc,
        'Baseline': base_acc,
        'F1': f1,
        'y_true': le.inverse_transform(y_test),
        'y_pred': le.inverse_transform(preds),
        'classes': le.classes_
    }

# -----------------------------
# 4. PLOTTING
# -----------------------------
def plot_results(results_list):
    if not results_list: return
    
    # 1. Confusion Matrices
    for res in results_list:
        y_true = res['y_true']
        y_pred = res['y_pred']
        
        # Top 10 classes
        counts = Counter(y_true)
        top_classes = [k for k,v in counts.most_common(10)]
        
        mask = np.isin(y_true, top_classes)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        cm = confusion_matrix(y_t, y_p, labels=top_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=top_classes, yticklabels=top_classes)
        plt.title(f"Confusion Matrix: {res['Target']} ({res['Embedding']})")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/cm_{res['Target']}_{res['Embedding']}.png")
        plt.close()

    # 2. Performance Bar Chart
    df = pd.DataFrame(results_list)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['Baseline'], width, label='Baseline', alpha=0.6)
    plt.bar(x + width/2, df['Accuracy'], width, label='KG Model', alpha=0.8)
    
    plt.xticks(x, [f"{r['Target']}\n({r['Embedding']})" for r in results_list], rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Structural Property Imputation Performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/performance_comparison.png")
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
def main():
    # 1. Get Labels
    df_labels = get_structural_labels_from_kg()
    if df_labels.empty: return
    
    # 2. Run for each embedding type
    results = []
    
    # Node2Vec
    df_n2v = load_embeddings(N2V_PATH)
    if df_n2v is not None:
        res_topo = run_imputation("topology", df_labels, df_n2v, "Node2Vec")
        if res_topo: results.append(res_topo)
        
        res_metal = run_imputation("metal_element", df_labels, df_n2v, "Node2Vec")
        if res_metal: results.append(res_metal)

    # GraphGPS
    df_gps = load_embeddings(GPS_PATH)
    if df_gps is not None:
        res_topo = run_imputation("topology", df_labels, df_gps, "GraphGPS")
        if res_topo: results.append(res_topo)
        
        res_metal = run_imputation("metal_element", df_labels, df_gps, "GraphGPS")
        if res_metal: results.append(res_metal)
        
    # Save raw metrics
    if results:
        pd.DataFrame([{k:v for k,v in r.items() if k not in ['y_true', 'y_pred', 'classes']} for r in results])\
          .to_csv(f"{OUTPUT_DIR}/imputation_metrics.csv", index=False)
        plot_results(results)
        logging.info("Imputation study complete.")
    else:
        logging.warning("No results generated.")

if __name__ == "__main__":
    logging.warning(
        "This legacy script is superseded by imputation_compare.py; "
        "running the new comparison pipeline now."
    )
    try:
        from imputation_compare import main as compare_main
        compare_main()
    except Exception:
        main()