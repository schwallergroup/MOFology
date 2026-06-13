#!/usr/bin/env python3
"""
Remake the combined figure using SAVED prediction data.
No re-running of models!
"""
import os
import numpy as np
import pandas as pd
import pickle

# First, save the prediction data if not already saved
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE, "paper", "figures", "generated", "compgcn_predictions_data.pkl")

# Check if data exists
if os.path.exists(DATA_FILE):
    print(f"Loading saved prediction data from {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    y_co2_actual = data['y_co2_actual']
    y_co2_pred = data['y_co2_pred']
    y_h2o_actual = data['y_h2o_actual']
    y_h2o_pred = data['y_h2o_pred']
    print(f"  Loaded: {len(y_co2_actual)} CO2 predictions, {len(y_h2o_actual)} H2O predictions")
else:
    print("ERROR: No saved prediction data found!")
    print(f"Expected file: {DATA_FILE}")
    print("Need to run the prediction script first to save the data.")
    exit(1)

# Now create the figure with updated styling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_theme(style="white")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

PLASMA = plt.cm.plasma
OUT_DIR = os.path.join(BASE, "paper", "figures", "generated")

print("Creating updated figure...")
fig = plt.figure(figsize=(10, 16))
gs = fig.add_gridspec(2, 1, wspace=0.0, hspace=0.20, left=0.12, right=0.85, top=0.92, bottom=0.06)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

# Calculate errors
err_co2 = np.abs(y_co2_actual - y_co2_pred)
err_h2o = np.abs(y_h2o_actual - y_h2o_pred)
all_errors = np.concatenate([err_co2, err_h2o])
norm = Normalize(vmin=all_errors.min(), vmax=all_errors.max())

# Panel (a): CO2
colors_co2 = PLASMA(norm(err_co2))
scatter1 = ax1.scatter(y_co2_actual, y_co2_pred, c=err_co2, cmap=PLASMA, 
                      vmin=all_errors.min(), vmax=all_errors.max(),
                      s=65, alpha=0.7, edgecolors='white', linewidths=0.6, rasterized=True)
lims_co2 = [min(y_co2_actual.min(), y_co2_pred.min()), 
            max(y_co2_actual.max(), y_co2_pred.max())]
ax1.plot(lims_co2, lims_co2, 'k--', alpha=0.6, linewidth=2.5, label='Perfect prediction')
r2_co2 = r2_score(y_co2_actual, y_co2_pred)
rmse_co2 = np.sqrt(mean_squared_error(y_co2_actual, y_co2_pred))
mae_co2 = mean_absolute_error(y_co2_actual, y_co2_pred)
ax1.text(0.05, 0.95, f"R² = {r2_co2:.3f}\nRMSE = {rmse_co2:.3f} eV\nMAE = {mae_co2:.3f} eV\nN = {len(y_co2_actual)}",
        transform=ax1.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=2))
ax1.set_xlabel("Actual ΔCO₂ binding energy (eV)", fontsize=20, fontweight='bold')
ax1.set_ylabel("Predicted ΔCO₂ binding energy (eV)", fontsize=20, fontweight='bold')
ax1.set_title("(a) ΔCO₂ binding energy", fontsize=22, fontweight='bold', pad=20)
ax1.legend(loc='lower right', fontsize=16, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax1.set_aspect('equal', adjustable='box')

# Panel (b): H2O
scatter2 = ax2.scatter(y_h2o_actual, y_h2o_pred, c=err_h2o, cmap=PLASMA,
                      vmin=all_errors.min(), vmax=all_errors.max(),
                      s=65, alpha=0.7, edgecolors='white', linewidths=0.6, rasterized=True)
lims_h2o = [min(y_h2o_actual.min(), y_h2o_pred.min()), 
            max(y_h2o_actual.max(), y_h2o_pred.max())]
ax2.plot(lims_h2o, lims_h2o, 'k--', alpha=0.6, linewidth=2.5, label='Perfect prediction')
r2_h2o = r2_score(y_h2o_actual, y_h2o_pred)
rmse_h2o = np.sqrt(mean_squared_error(y_h2o_actual, y_h2o_pred))
mae_h2o = mean_absolute_error(y_h2o_actual, y_h2o_pred)
ax2.text(0.05, 0.95, f"R² = {r2_h2o:.3f}\nRMSE = {rmse_h2o:.3f} eV\nMAE = {mae_h2o:.3f} eV\nN = {len(y_h2o_actual)}",
        transform=ax2.transAxes, fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=2))
ax2.set_xlabel("Actual ΔH₂O binding energy (eV)", fontsize=20, fontweight='bold')
ax2.set_ylabel("Predicted ΔH₂O binding energy (eV)", fontsize=20, fontweight='bold')
ax2.set_title("(b) ΔH₂O binding energy", fontsize=22, fontweight='bold', pad=20)
ax2.legend(loc='lower right', fontsize=16, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax2.set_aspect('equal', adjustable='box')

# Shared colorbar - positioned to the right of both subplots
cbar_ax = fig.add_axes([0.88, 0.06, 0.02, 0.88])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter2, cax=cbar_ax)
cbar.set_label('Absolute Error (eV)', fontsize=20, fontweight='bold', labelpad=20)
cbar.ax.tick_params(labelsize=16)

# NEW TITLE
fig.suptitle("KG Embedding Prediction on Functionalization Effect",
            fontsize=26, fontweight='bold', y=0.98)

out_path = os.path.join(OUT_DIR, "fig_compgcn_kg_predictions_combined.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Updated: {out_path}")

plt.close(fig)
print("Done!")
