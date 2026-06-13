#!/usr/bin/env python3
"""
Create publication-quality combined figure for CompGCN KG predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Publication settings
sns.set_theme(style="white")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "paper", "figures", "generated")
PLASMA = plt.cm.plasma

# Load the saved metrics to get the predictions
metrics_df = pd.read_csv(os.path.join(OUT_DIR, "compgcn_kg_metrics_by_amine.csv"))

# We need to regenerate the predictions since they weren't saved
# Load from the script output or recreate them
print("Loading prediction data...")

# For now, let's read the data and recreate
import sys
sys.path.insert(0, os.path.join(BASE, "scripts"))

# Actually, let me just load and use the existing combined plot data
# by re-running a minimal version

from rdflib import Graph
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False

HPC = os.path.join(BASE, "MOFKG_from_hpc")
KG_PATH = os.path.join(HPC, "KG", "data", "KG", "mof_kg.ttl")
COMPGCN_CSV = os.path.join(HPC, "studies", "data", "gnn_embeddings",
                           "mof_compgcn_embeddings_256d_3layers.csv")
SEED = 42
CV_FOLDS = 5

def get_parent_child_pairs(kg_path):
    g = Graph()
    g.parse(kg_path, format="turtle")
    query = """
    PREFIX mof: <http://emmo.info/domain-mof/mof-ontology#>
    PREFIX syn: <http://emmo.info/domain-mof/synthesis#>
    SELECT DISTINCT ?parentMof ?funcMof ?propMof ?amineCode ?funcCO2BE ?parentCO2BE ?funcH2OBE ?parentH2OBE
    WHERE {
      ?funcMof syn:hasFunctionalization ?func .
      ?func syn:hasFunctionalizationType syn:AmineFunctionalization .
      OPTIONAL { ?func syn:usesFunctionalGroup ?amineCode . }
      ?funcMof syn:derivedFrom ?parentMof .
      BIND(IRI(REPLACE(STR(?funcMof), "FuncMOF_", "MOF_")) AS ?propMof)
      OPTIONAL {
        { ?p1 mof:hasComputationalPropertyOwner ?propMof . }
        UNION { ?propMof mof:hasComputationalProperty ?p1 . }
        ?p1 mof:propertyName ?n1 ; mof:propertyValue ?funcCO2BE .
        FILTER(CONTAINS(LCASE(STR(?n1)), "co2") && 
               (CONTAINS(LCASE(STR(?n1)), "binding energy") || CONTAINS(LCASE(STR(?n1)), "binding")))
      }
      OPTIONAL {
        { ?p2 mof:hasComputationalPropertyOwner ?parentMof . }
        UNION { ?parentMof mof:hasComputationalProperty ?p2 . }
        ?p2 mof:propertyName ?n2 ; mof:propertyValue ?parentCO2BE .
        FILTER(CONTAINS(LCASE(STR(?n2)), "co2") && 
               (CONTAINS(LCASE(STR(?n2)), "binding energy") || CONTAINS(LCASE(STR(?n2)), "binding")))
      }
      OPTIONAL {
        { ?p3 mof:hasComputationalPropertyOwner ?propMof . }
        UNION { ?propMof mof:hasComputationalProperty ?p3 . }
        ?p3 mof:propertyName ?n3 ; mof:propertyValue ?funcH2OBE .
        FILTER(CONTAINS(LCASE(STR(?n3)), "h2o") && 
               (CONTAINS(LCASE(STR(?n3)), "binding energy") || CONTAINS(LCASE(STR(?n3)), "binding")))
      }
      OPTIONAL {
        { ?p4 mof:hasComputationalPropertyOwner ?parentMof . }
        UNION { ?parentMof mof:hasComputationalProperty ?p4 . }
        ?p4 mof:propertyName ?n4 ; mof:propertyValue ?parentH2OBE .
        FILTER(CONTAINS(LCASE(STR(?n4)), "h2o") && 
               (CONTAINS(LCASE(STR(?n4)), "binding energy") || CONTAINS(LCASE(STR(?n4)), "binding")))
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
            "parent_co2_be": float(row.parentCO2BE),
            "child_co2_be": float(row.funcCO2BE),
            "parent_h2o_be": float(row.parentH2OBE),
            "child_h2o_be": float(row.funcH2OBE),
        })
    df = pd.DataFrame(rows)
    df["Delta_CO2"] = df["child_co2_be"] - df["parent_co2_be"]
    df["Delta_H2O"] = df["child_h2o_be"] - df["parent_h2o_be"]
    return df

def load_embeddings():
    df = pd.read_csv(COMPGCN_CSV)
    if "mof_uri" not in df.columns:
        df = df.rename(columns={df.columns[0]: "mof_uri"})
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    return df, emb_cols

def create_features(pairs_df, emb_df, emb_cols):
    emb_indexed = emb_df.set_index("mof_uri")
    X_list, valid_idx = [], []
    for idx, row in pairs_df.iterrows():
        if row["parent_uri"] in emb_indexed.index and row["child_uri"] in emb_indexed.index:
            parent_emb = emb_indexed.loc[row["parent_uri"], emb_cols].values
            child_emb = emb_indexed.loc[row["child_uri"], emb_cols].values
            X_list.append(np.concatenate([parent_emb, child_emb]))
            valid_idx.append(idx)
    return np.vstack(X_list).astype(np.float32), pairs_df.loc[valid_idx].reset_index(drop=True)

def train_and_predict(X, y, cv_folds=5, seed=42):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    all_pred, all_actual = [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if HAS_XGB:
            model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=seed, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10,
                                         random_state=seed, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        all_pred.extend(model.predict(X_test_scaled))
        all_actual.extend(y_test)
    return np.array(all_actual), np.array(all_pred)

print("Loading data...")
pairs = get_parent_child_pairs(KG_PATH)
emb_df, emb_cols = load_embeddings()
X, pairs_sub = create_features(pairs, emb_df, emb_cols)

print("Running predictions...")
y_co2_actual, y_co2_pred = train_and_predict(X, pairs_sub["Delta_CO2"].values, CV_FOLDS, SEED)
y_h2o_actual, y_h2o_pred = train_and_predict(X, pairs_sub["Delta_H2O"].values, CV_FOLDS, SEED)

# Create publication figure
print("Creating publication figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Calculate errors for coloring
err_co2 = np.abs(y_co2_actual - y_co2_pred)
err_h2o = np.abs(y_h2o_actual - y_h2o_pred)

# Use combined error range for consistent colorbar
all_errors = np.concatenate([err_co2, err_h2o])
norm = Normalize(vmin=all_errors.min(), vmax=all_errors.max())

# Panel (a): CO2
colors_co2 = PLASMA(norm(err_co2))
scatter1 = ax1.scatter(y_co2_actual, y_co2_pred, c=colors_co2, s=50, alpha=0.7,
                      edgecolors='white', linewidths=0.5, rasterized=True)
lims_co2 = [min(y_co2_actual.min(), y_co2_pred.min()), 
            max(y_co2_actual.max(), y_co2_pred.max())]
ax1.plot(lims_co2, lims_co2, 'k--', alpha=0.6, linewidth=2, label='Perfect prediction')
r2_co2 = r2_score(y_co2_actual, y_co2_pred)
rmse_co2 = np.sqrt(mean_squared_error(y_co2_actual, y_co2_pred))
mae_co2 = mean_absolute_error(y_co2_actual, y_co2_pred)
ax1.text(0.05, 0.95, f"R² = {r2_co2:.3f}\nRMSE = {rmse_co2:.3f} eV\nMAE = {mae_co2:.3f} eV\nN = {len(y_co2_actual)}",
        transform=ax1.transAxes, fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
ax1.set_xlabel("Actual ΔCO₂ binding energy (eV)", fontsize=16, fontweight='bold')
ax1.set_ylabel("Predicted ΔCO₂ binding energy (eV)", fontsize=16, fontweight='bold')
ax1.set_title("(a) ΔCO₂ Binding Energy", fontsize=18, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_aspect('equal', adjustable='box')

# Panel (b): H2O
colors_h2o = PLASMA(norm(err_h2o))
scatter2 = ax2.scatter(y_h2o_actual, y_h2o_pred, c=colors_h2o, s=50, alpha=0.7,
                      edgecolors='white', linewidths=0.5, rasterized=True)
lims_h2o = [min(y_h2o_actual.min(), y_h2o_pred.min()), 
            max(y_h2o_actual.max(), y_h2o_pred.max())]
ax2.plot(lims_h2o, lims_h2o, 'k--', alpha=0.6, linewidth=2, label='Perfect prediction')
r2_h2o = r2_score(y_h2o_actual, y_h2o_pred)
rmse_h2o = np.sqrt(mean_squared_error(y_h2o_actual, y_h2o_pred))
mae_h2o = mean_absolute_error(y_h2o_actual, y_h2o_pred)
ax2.text(0.05, 0.95, f"R² = {r2_h2o:.3f}\nRMSE = {rmse_h2o:.3f} eV\nMAE = {mae_h2o:.3f} eV\nN = {len(y_h2o_actual)}",
        transform=ax2.transAxes, fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
ax2.set_xlabel("Actual ΔH₂O binding energy (eV)", fontsize=16, fontweight='bold')
ax2.set_ylabel("Predicted ΔH₂O binding energy (eV)", fontsize=16, fontweight='bold')
ax2.set_title("(b) ΔH₂O Binding Energy", fontsize=18, fontweight='bold', pad=15)
ax2.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_aspect('equal', adjustable='box')

# Shared colorbar on the right
cbar = fig.colorbar(scatter2, ax=[ax1, ax2], fraction=0.03, pad=0.02, aspect=30)
cbar.set_label('Absolute Error (eV)', fontsize=16, fontweight='bold', labelpad=15)
cbar.ax.tick_params(labelsize=13)

fig.suptitle("CompGCN (KG Embeddings): Amine Functionalization Effect Prediction",
            fontsize=20, fontweight='bold', y=0.98)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "fig_publication_compgcn_predictions.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")

# Also save as PDF for publications
pdf_path = out_path.replace('.png', '.pdf')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"Saved: {pdf_path}")

plt.close(fig)
print("Done!")
