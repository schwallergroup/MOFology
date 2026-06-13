import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "chemcial_properties.csv")
EMB_PATHS = {
    "CompGCN": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_compgcn_embeddings_256d_2layers.csv),
    "Node2Vec": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_p1.0_q1.0.csv)
}
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "results"/ML_Chem/gnn_n2v_results)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_embedding(name, df_prop, df_emb):
    logging.info(f"Evaluating {name} embeddings...")
    
    # Merge on mof_uri
    df = pd.merge(df_prop, df_emb, on='mof_uri', how='inner')
    logging.info(f"Merged data size for {name}: {df.shape}")
    
    # Identify targets and embedding columns
    metadata_cols = ['mof_uri', 'csd_code', 'chemical_formula', 'mofid', 
                     'topology', 'metal_cluster_elements', 'linker_smiles', 
                     'space_group', 'crystal_system']
    
    # Embedding columns start with emb_
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    # Target columns are numeric and not metadata or embeddings
    target_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                   if c not in metadata_cols and c not in emb_cols]
    
    results = []
    
    for target in target_cols:
        y = df[target]
        mask = y.notna()
        if mask.sum() < 100:
            continue
            
        logging.info(f"  Target: {target} ({mask.sum()} samples)")
        
        X = df.loc[mask, emb_cols]
        y = y[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models = {
            "Ridge": Ridge(),
            "RandomForest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
            "XGBoost": XGBRegressor(n_jobs=-1),
            "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(512, 256), max_iter=1000, early_stopping=True)
        }
        
        for m_name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            results.append({
                "Target": target,
                "Model": m_name,
                "R2": r2,
                "RMSE": rmse,
                "Embedding": name
            })
            
    return results

def main():
    logging.info("Loading chemical properties...")
    df_prop = pd.read_csv(DATA_PATH)
    
    all_results = []
    for name, path in EMB_PATHS.items():
        if not os.path.exists(path):
            logging.warning(f"Embedding file not found: {path}")
            continue
        df_emb = pd.read_csv(path)
        results = evaluate_embedding(name, df_prop, df_emb)
        all_results.extend(results)
        
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{OUTPUT_DIR}/gnn_n2v_chem_results.csv", index=False)
    
    # Visualization: Heatmap Comparison
    # We'll create one heatmap per embedding type for clarity
    for name in EMB_PATHS.keys():
        subset = results_df[results_df['Embedding'] == name]
        if subset.empty: continue
        
        pivot_df = subset.pivot(index="Target", columns="Model", values="R2")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"{name} Embeddings: Chemical Property Prediction (R2)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{name.lower()}_chem_heatmap.png")
        
    logging.info(f"Study complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
