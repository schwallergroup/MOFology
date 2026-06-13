#!/usr/bin/env python3
"""
Generate predicted vs actual scatter plots for CompGCN KG embeddings
on Delta_CO2 and Delta_H2O predictions, colored by error.
Also generate a table of metrics by amine type.

FAST VERSION - Only runs CompGCN KG predictions (5 folds x 2 targets)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from rdflib import Graph
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

# Configuration
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPC = os.path.join(BASE, "MOFKG_from_hpc")
KG_PATH = os.path.join(HPC, "KG", "data", "KG", "mof_kg.ttl")
COMPGCN_CSV = os.path.join(HPC, "studies", "data", "gnn_embeddings",
                           "mof_compgcn_embeddings_256d_3layers.csv")
OUT_DIR = os.path.join(BASE, "paper", "figures", "generated")

SEED = 42
CV_FOLDS = 5
PLASMA = plt.cm.plasma

print(f"Using KG path: {KG_PATH}")
print(f"Using embeddings: {COMPGCN_CSV}")

# Styling
sns.set_theme(style="whitegrid", palette="plasma")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("WARNING: XGBoost not available, using RandomForest")

os.makedirs(OUT_DIR, exist_ok=True)


def get_parent_child_pairs(kg_path: str) -> pd.DataFrame:
    """Extract parent-child functionalization pairs from KG - copied from working script."""
    print("Querying knowledge graph for functionalization pairs...")
    g = Graph()
    g.parse(kg_path, format="turtle")
    
    query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    PREFIX syn: <http://emmo.info/domain-mof/synthesis#>
    SELECT DISTINCT ?parentMof ?funcMof ?propMof ?amineCode ?amineName ?parentCO2BE ?funcCO2BE ?parentH2OBE ?funcH2OBE
    WHERE {
      ?funcMof syn:hasFunctionalization ?func .
      ?func syn:hasFunctionalizationType syn:AmineFunctionalization .
      OPTIONAL { ?func syn:usesFunctionalGroup ?amineCode . }
      OPTIONAL { ?func syn:functionalGroupName ?amineName . }
      ?funcMof syn:derivedFrom ?parentMof .
      BIND(IRI(REPLACE(STR(?funcMof), "FuncMOF_", "MOF_")) AS ?propMof)
      OPTIONAL {
        {
          ?p1 mof:hasComputationalPropertyOwner ?propMof .
        } UNION {
          ?propMof mof:hasComputationalProperty ?p1 .
        }
        ?p1 mof:propertyName ?n1 ;
            mof:propertyValue ?funcCO2BE .
        FILTER(
          CONTAINS(LCASE(STR(?n1)), "co2") &&
          (CONTAINS(LCASE(STR(?n1)), "binding energy") || CONTAINS(LCASE(STR(?n1)), "binding"))
        )
      }
      OPTIONAL {
        {
          ?p2 mof:hasComputationalPropertyOwner ?parentMof .
        } UNION {
          ?parentMof mof:hasComputationalProperty ?p2 .
        }
        ?p2 mof:propertyName ?n2 ;
            mof:propertyValue ?parentCO2BE .
        FILTER(
          CONTAINS(LCASE(STR(?n2)), "co2") &&
          (CONTAINS(LCASE(STR(?n2)), "binding energy") || CONTAINS(LCASE(STR(?n2)), "binding"))
        )
      }
      OPTIONAL {
        {
          ?p3 mof:hasComputationalPropertyOwner ?propMof .
        } UNION {
          ?propMof mof:hasComputationalProperty ?p3 .
        }
        ?p3 mof:propertyName ?n3 ;
            mof:propertyValue ?funcH2OBE .
        FILTER(
          CONTAINS(LCASE(STR(?n3)), "h2o") &&
          (CONTAINS(LCASE(STR(?n3)), "binding energy") || CONTAINS(LCASE(STR(?n3)), "binding"))
        )
      }
      OPTIONAL {
        {
          ?p4 mof:hasComputationalPropertyOwner ?parentMof .
        } UNION {
          ?parentMof mof:hasComputationalProperty ?p4 .
        }
        ?p4 mof:propertyName ?n4 ;
            mof:propertyValue ?parentH2OBE .
        FILTER(
          CONTAINS(LCASE(STR(?n4)), "h2o") &&
          (CONTAINS(LCASE(STR(?n4)), "binding energy") || CONTAINS(LCASE(STR(?n4)), "binding"))
        )
      }
    }
    """
    
    results = g.query(query)
    rows = []
    for row in results:
        if row.parentCO2BE is None or row.funcCO2BE is None:
            continue
        if row.parentH2OBE is None or row.funcH2OBE is None:
            continue
        rows.append({
            "parent_uri": str(row.parentMof),
            "child_uri": str(row.funcMof),
            "amine_code": str(row.amineCode) if row.amineCode else "unknown",
            "amine_name": str(row.amineName) if row.amineName else "unknown",
            "parent_co2_be": float(row.parentCO2BE),
            "child_co2_be": float(row.funcCO2BE),
            "parent_h2o_be": float(row.parentH2OBE),
            "child_h2o_be": float(row.funcH2OBE),
        })
    
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No functionalization pairs found in KG!")
    
    df["Delta_CO2"] = df["child_co2_be"] - df["parent_co2_be"]
    df["Delta_H2O"] = df["child_h2o_be"] - df["parent_h2o_be"]
    
    print(f"  Found {len(df)} functionalization pairs")
    print(f"  Amine types: {sorted(df['amine_code'].value_counts().to_dict().items())}")
    
    return df


def load_compgcn_embeddings() -> pd.DataFrame:
    """Load CompGCN embeddings."""
    print(f"Loading CompGCN embeddings from {COMPGCN_CSV}...")
    df = pd.read_csv(COMPGCN_CSV)
    if "mof_uri" not in df.columns:
        df = df.rename(columns={df.columns[0]: "mof_uri"})
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        emb_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  Loaded {len(df)} entities with {len(emb_cols)}D embeddings")
    return df


def create_kg_features(pairs_df, emb_df):
    """Create KG feature matrix by concatenating parent and child embeddings."""
    print("Creating KG feature matrix...")
    
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    emb_df_indexed = emb_df.set_index("mof_uri")
    
    # For each pair, concatenate parent and child embeddings
    X_list = []
    valid_idx = []
    
    for idx, row in pairs_df.iterrows():
        parent_uri = row["parent_uri"]
        child_uri = row["child_uri"]
        
        if parent_uri in emb_df_indexed.index and child_uri in emb_df_indexed.index:
            parent_emb = emb_df_indexed.loc[parent_uri, emb_cols].values
            child_emb = emb_df_indexed.loc[child_uri, emb_cols].values
            X_list.append(np.concatenate([parent_emb, child_emb]))
            valid_idx.append(idx)
    
    X = np.vstack(X_list).astype(np.float32)
    pairs_subset = pairs_df.loc[valid_idx].reset_index(drop=True)
    
    print(f"  Created feature matrix: {X.shape} for {len(pairs_subset)} pairs")
    return X, pairs_subset


def train_and_predict(X, y, amine_codes, cv_folds=5, seed=42):
    """Train model with cross-validation and return all predictions."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    all_predictions = []
    all_actuals = []
    all_amine_codes = []
    all_fold_ids = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        amine_test = amine_codes[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if HAS_XGB:
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                random_state=seed, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                         random_state=seed, n_jobs=-1)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        all_amine_codes.extend(amine_test)
        all_fold_ids.extend([fold_idx] * len(y_test))
    
    return np.array(all_actuals), np.array(all_predictions), np.array(all_amine_codes), np.array(all_fold_ids)


def plot_predictions(y_true, y_pred, target_name, out_path):
    """Create predicted vs actual scatter plot colored by error."""
    print(f"  Creating scatter plot for {target_name}...")
    
    errors = np.abs(y_true - y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6.5))
    
    # Normalize errors for coloring
    norm = Normalize(vmin=errors.min(), vmax=errors.max())
    colors = PLASMA(norm(errors))
    
    # Scatter plot
    scatter = ax.scatter(y_true, y_pred, c=colors, s=25, alpha=0.7,
                        edgecolors='white', linewidths=0.3, rasterized=True)
    
    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect prediction')
    
    # Metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Add metrics text
    metrics_text = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nN = {len(y_true)}"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and title
    target_label = target_name.replace("_", " ")
    ax.set_xlabel(f"Actual {target_label} (eV)", fontsize=11)
    ax.set_ylabel(f"Predicted {target_label} (eV)", fontsize=11)
    ax.set_title(f"CompGCN (KG Embedding): {target_label} Prediction",
                fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error (eV)', fontsize=10)
    
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def create_combined_plot(data_dict, out_path):
    """Create 2-panel plot with CO2 and H2O predictions side by side."""
    print("\nCreating combined 2-panel plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, (target, data) in zip([ax1, ax2], data_dict.items()):
        y_true = data['actual']
        y_pred = data['predicted']
        errors = np.abs(y_true - y_pred)
        
        # Normalize errors for coloring
        norm = Normalize(vmin=errors.min(), vmax=errors.max())
        colors = PLASMA(norm(errors))
        
        # Scatter plot
        scatter = ax.scatter(y_true, y_pred, c=colors, s=25, alpha=0.7,
                           edgecolors='white', linewidths=0.3, rasterized=True)
        
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='Perfect prediction')
        
        # Metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Add metrics text
        metrics_text = f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nN = {len(y_true)}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        # Labels and title
        panel_label = "(a)" if target == "Delta_CO2" else "(b)"
        target_label = "ΔCO₂ binding energy" if target == "Delta_CO2" else "ΔH₂O binding energy"
        ax.set_xlabel(f"Actual {target_label} (eV)", fontsize=11)
        ax.set_ylabel(f"Predicted {target_label} (eV)", fontsize=11)
        ax.set_title(f"{panel_label} {target_label}", fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Absolute Error (eV)', fontsize=9)
        
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    fig.suptitle("CompGCN (KG Embeddings Only): Functionalization Effect Prediction",
                fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def create_amine_metrics_table(data_dict, out_csv):
    """Create table of metrics broken down by amine type and fold."""
    print("\nCreating amine type metrics table...")
    
    rows = []
    
    for target, data in data_dict.items():
        y_true = data['actual']
        y_pred = data['predicted']
        amine_codes = data['amine_codes']
        fold_ids = data['fold_ids']
        
        # Overall metrics (ALL)
        for fold in range(CV_FOLDS):
            mask = fold_ids == fold
            if mask.sum() > 0:
                y_t = y_true[mask]
                y_p = y_pred[mask]
                rows.append({
                    'Target': target,
                    'AmineType': 'ALL',
                    'Fold': fold,
                    'N': len(y_t),
                    'R2': r2_score(y_t, y_p),
                    'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
                    'MAE': mean_absolute_error(y_t, y_p),
                })
        
        # Per amine type metrics
        for amine in np.unique(amine_codes):
            if str(amine) == 'unknown':
                continue
            
            for fold in range(CV_FOLDS):
                mask = (amine_codes == amine) & (fold_ids == fold)
                if mask.sum() >= 3:  # At least 3 samples
                    y_t = y_true[mask]
                    y_p = y_pred[mask]
                    rows.append({
                        'Target': target,
                        'AmineType': str(amine),
                        'Fold': fold,
                        'N': len(y_t),
                        'R2': r2_score(y_t, y_p),
                        'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
                        'MAE': mean_absolute_error(y_t, y_p),
                    })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv}")
    
    # Create summary table (aggregated across folds)
    summary_rows = []
    for target in ['Delta_CO2', 'Delta_H2O']:
        for amine in df['AmineType'].unique():
            subset = df[(df['Target'] == target) & (df['AmineType'] == amine)]
            if len(subset) > 0:
                summary_rows.append({
                    'Target': target,
                    'AmineType': amine,
                    'N_total': subset['N'].sum(),
                    'R2_mean': subset['R2'].mean(),
                    'R2_std': subset['R2'].std(),
                    'RMSE_mean': subset['RMSE'].mean(),
                    'RMSE_std': subset['RMSE'].std(),
                    'MAE_mean': subset['MAE'].mean(),
                    'MAE_std': subset['MAE'].std(),
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_csv.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved {summary_csv}")
    
    return df, summary_df


def main():
    print("=" * 70)
    print("CompGCN KG Embedding: Functionalization Prediction Analysis")
    print("=" * 70)
    
    # Load data
    pairs_df = get_parent_child_pairs(KG_PATH)
    emb_df = load_compgcn_embeddings()
    
    # Create features
    X, pairs_subset = create_kg_features(pairs_df, emb_df)
    
    # Prepare targets
    data_dict = {}
    
    for target in ['Delta_CO2', 'Delta_H2O']:
        print(f"\n{'=' * 70}")
        print(f"Training model for {target}")
        print('=' * 70)
        
        y = pairs_subset[target].values
        amine_codes = pairs_subset['amine_code'].values
        
        # Train and get predictions
        y_true, y_pred, amine_codes_out, fold_ids = train_and_predict(
            X, y, amine_codes, cv_folds=CV_FOLDS, seed=SEED
        )
        
        data_dict[target] = {
            'actual': y_true,
            'predicted': y_pred,
            'amine_codes': amine_codes_out,
            'fold_ids': fold_ids
        }
        
        # Create individual plot
        individual_path = os.path.join(OUT_DIR, f"compgcn_kg_{target.lower()}_predictions.png")
        plot_predictions(y_true, y_pred, target, individual_path)
    
    # Create combined 2-panel plot
    combined_path = os.path.join(OUT_DIR, "fig_compgcn_kg_predictions_combined.png")
    create_combined_plot(data_dict, combined_path)
    
    # Create metrics tables
    metrics_csv = os.path.join(OUT_DIR, "compgcn_kg_metrics_by_amine.csv")
    df_metrics, df_summary = create_amine_metrics_table(data_dict, metrics_csv)
    
    print("\n" + "=" * 70)
    print("Summary Statistics by Amine Type")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    # SAVE THE RAW PREDICTION DATA FOR REUSE
    import pickle
    data_file = os.path.join(OUT_DIR, "compgcn_predictions_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump({
            'y_co2_actual': data_dict['Delta_CO2']['actual'],
            'y_co2_pred': data_dict['Delta_CO2']['predicted'],
            'y_h2o_actual': data_dict['Delta_H2O']['actual'],
            'y_h2o_pred': data_dict['Delta_H2O']['predicted'],
        }, f)
    print(f"  - Saved prediction data: {data_file}")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Combined plot: {combined_path}")
    print(f"  - Individual plots: {OUT_DIR}/compgcn_kg_delta_*.png")
    print(f"  - Metrics by fold: {metrics_csv}")
    print(f"  - Summary by amine: {metrics_csv.replace('.csv', '_summary.csv')}")
    print(f"  - Prediction data (for reuse): {data_file}")


if __name__ == "__main__":
    main()
