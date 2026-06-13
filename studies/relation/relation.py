import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from rdflib import Graph

# -----------------------------
# CONFIG
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kg", "mof_kg.ttl")
N2V_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "node2vec"/mof_embeddings_p1.0_q1.0.csv)
GPS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings", "data", "gnn_embeddings"/mof_advanced_embeddings_256d_4layers_1000epochs.csv)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "studies", "data", "relation")

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# MULTI-TARGET MODEL
# -----------------------------
class MOFDeltaPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Shared backbone to learn general MOF features
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # Separate heads for CO2 and H2O deltas
        self.co2_head = nn.Linear(256, 1)
        self.h2o_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.backbone(x)
        co2_delta = self.co2_head(features)
        h2o_delta = self.h2o_head(features)
        return torch.cat((co2_delta, h2o_delta), dim=1) # Shape: [batch, 2]

# -----------------------------
# DATA EXTRACTION
# -----------------------------
def get_data_from_kg():
    g = Graph()
    logging.info(f"Loading KG from {KG_PATH}...")
    try:
        g.parse(KG_PATH, format="turtle")
    except Exception as e:
        logging.error(f"Failed to load KG: {e}")
        return pd.DataFrame()
    
    # UPDATED QUERY: Bridges FuncMOF_ -> MOF_ for property lookup
    query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    PREFIX syn: <http://emmo.info/domain-mof/synthesis#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?parentMof ?funcMof ?parentCO2BE ?funcCO2BE ?parentH2OBE ?funcH2OBE
    WHERE {
      # 1. Identify the Functionalized MOF (Semantic Node)
      ?funcMof syn:hasFunctionalization ?func .
      ?func syn:hasFunctionalizationType syn:AmineFunctionalization .
      ?funcMof syn:derivedFrom ?parentMof .

      # 2. BRIDGE: Create the Property Node URI by replacing 'FuncMOF_' with 'MOF_'
      # This connects the structural definition to the node containing the calculated properties
      BIND(IRI(REPLACE(STR(?funcMof), "FuncMOF_", "MOF_")) AS ?propMof)

      # 3. Fetch CO2 Properties
      # Use ?propMof for the child (functionalized) values
      OPTIONAL {
        ?propMof mof:hasComputationalProperty ?p1 .
        ?p1 mof:propertyName "CO2 binding energy"^^xsd:string ; mof:propertyValue ?funcCO2BE .
      }
      # Use ?parentMof for the parent values
      OPTIONAL {
        ?parentMof mof:hasComputationalProperty ?p2 .
        ?p2 mof:propertyName "CO2 binding energy"^^xsd:string ; mof:propertyValue ?parentCO2BE .
      }
      
      # 4. Fetch H2O Properties
      OPTIONAL {
        ?propMof mof:hasComputationalProperty ?p3 .
        ?p3 mof:propertyName "H2O binding energy"^^xsd:string ; mof:propertyValue ?funcH2OBE .
      }
      OPTIONAL {
        ?parentMof mof:hasComputationalProperty ?p4 .
        ?p4 mof:propertyName "H2O binding energy"^^xsd:string ; mof:propertyValue ?parentH2OBE .
      }
    }
    """
    
    results = []
    logging.info("Executing SPARQL query...")
    for row in g.query(query):
        results.append({
            'parent_uri': str(row.parentMof),
            'child_uri': str(row.funcMof),
            'p_co2': float(row.parentCO2BE) if row.parentCO2BE else np.nan,
            'c_co2': float(row.funcCO2BE) if row.funcCO2BE else np.nan,
            'p_h2o': float(row.parentH2OBE) if row.parentH2OBE else np.nan,
            'c_h2o': float(row.funcH2OBE) if row.funcH2OBE else np.nan
        })
    
    df = pd.DataFrame(results)
    
    # Debug info
    if not df.empty:
        logging.info(f"Found {len(df)} candidate pairs before filtering.")
        logging.info(f"Missing Values:\n{df.isna().sum()}")
    
    df = df.dropna()
    logging.info(f"Extracted {len(df)} complete pairs for training.")
    return df

# -----------------------------
# EXECUTION
# -----------------------------
def main():
    df_pairs = get_data_from_kg()
    if df_pairs.empty:
        logging.warning("No data found. Exiting.")
        return

    embeddings = {'node2vec': N2V_PATH, 'GraphGPS': GPS_PATH}

    for name, path in embeddings.items():
        if not os.path.exists(path):
            logging.warning(f"Embedding file not found: {path}")
            continue
            
        logging.info(f"Processing embeddings: {name}")
        emb_df = pd.read_csv(path).set_index('mof_uri')
        
        # Filter and calculate Deltas
        # Ensure we only use pairs where the parent has an embedding
        valid = df_pairs[df_pairs['parent_uri'].isin(emb_df.index)].copy()
        
        if valid.empty:
            logging.warning(f"No matching embeddings found for {name}. Skipping.")
            continue
            
        X = emb_df.loc[valid['parent_uri']].filter(like='emb_').values
        
        # Targets: [Delta CO2, Delta H2O]
        # Delta = Child - Parent
        y_co2 = (valid['c_co2'] - valid['p_co2']).values
        y_h2o = (valid['c_h2o'] - valid['p_h2o']).values
        Y = np.column_stack((y_co2, y_h2o))

        # Train
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        Y_t = torch.tensor(Y, dtype=torch.float32).to(DEVICE)
        
        model = MOFDeltaPredictor(X.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        logging.info(f"Training multi-target model for {name}...")
        for epoch in range(150):
            model.train()
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, Y_t)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logging.info(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        # Inference on all MOFs (Prediction for Hypothetical Functionalization)
        model.eval()
        with torch.no_grad():
            all_X = torch.tensor(emb_df.filter(like='emb_').values, dtype=torch.float32).to(DEVICE)
            all_preds = model(all_X).cpu().numpy()
        
        output = pd.DataFrame(all_preds, columns=['pred_delta_CO2', 'pred_delta_H2O'], index=emb_df.index)
        output_path = f"{OUTPUT_DIR}/{name}_dual_predictions.csv"
        output.to_csv(output_path)
        logging.info(f"Saved results to {output_path}")

if __name__ == "__main__":
    logging.warning(
        "This legacy script is superseded by relation_compare.py; "
        "running the new comparison pipeline now."
    )
    try:
        from relation_compare import main as compare_main
        compare_main()
    except Exception:
        main()