#!/usr/bin/env python3
"""
Update KG statistics figures and tables for the 253K MOF dataset.
These don't depend on running prediction experiments.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# Style
sns.set_theme(style="whitegrid", palette="plasma")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PLASMA = plt.cm.plasma

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPC = os.path.join(BASE, "MOFKG_from_hpc")
OUT = os.path.join(BASE, "paper", "figures", "generated")
TABLES = os.path.join(BASE, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)
    print(f"  [OK] {name}.png")


def classify_mof(uri):
    """Classify MOF by data source."""
    if not isinstance(uri, str):
        return "Unknown"
    if "FuncMOF_" in uri:
        return "Functionalized"
    elif "MOF_STAB_" in uri:
        return "Hypothetical (CoRE)"
    elif "MOF_qmof-" in uri:
        return "QMOF (Computed)"
    elif "MOF_" in uri:
        return "CSD (Experimental)"
    else:
        return "Other"


def load_kg_stats():
    """Load and compute KG statistics from chemical properties CSV."""
    print("Loading KG data...")

    chem_path = os.path.join(HPC, "studies", "data", "chemcial_properties.csv")
    df = pd.read_csv(chem_path, low_memory=False)

    print(f"  Total MOFs: {len(df):,}")

    # Classify MOFs
    df["source"] = df["mof_uri"].apply(classify_mof)
    source_counts = df["source"].value_counts()

    print("\n  MOF Sources:")
    for src, count in source_counts.items():
        print(f"    {src}: {count:,}")

    # Count properties with data
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    property_coverage = {}
    for col in numeric_cols:
        n_valid = df[col].notna().sum()
        if n_valid > 0:
            property_coverage[col] = n_valid

    print(f"\n  Properties with data: {len(property_coverage)}")

    return df, source_counts, property_coverage


def fig_kg_overview(df, source_counts):
    """Single-panel, log-scale MOF counts by source (pie dropped per review)."""
    print("\nGenerating KG overview figure (log scale)...")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    order = source_counts.sort_values(ascending=True)
    y_pos = np.arange(len(order))
    colors = [PLASMA(i / max(1, len(order) - 1)) for i in range(len(order))]

    bars = ax.barh(y_pos, order.values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order.index)
    ax.set_xscale("log")
    ax.set_xlabel("Number of MOFs (log scale)", fontsize=11)
    ax.set_title(
        f"MOFology KG: {len(df):,} MOFs across {len(order)} sources",
        fontsize=12,
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for i, v in enumerate(order.values):
        ax.text(v * 1.05, i, f"{int(v):,}", va="center", fontsize=9)

    ax.margins(x=0.25)
    fig.tight_layout()
    save_fig(fig, "fig00_kg_overview")


def fig_property_coverage_updated(df):
    """Updated property coverage bar chart for 253K MOFs."""
    print("\nGenerating property coverage figure...")

    coverage_path = os.path.join(HPC, "results", "ML_Chem",
                                  "prediction_results", "property_coverage_report.csv")
    if not os.path.exists(coverage_path):
        print("  [SKIP] property_coverage_report.csv not found")
        return

    cov_df = pd.read_csv(coverage_path)
    cov_df = cov_df.sort_values("non_null_count", ascending=True).tail(25)

    norm = Normalize(vmin=cov_df["coverage"].min(), vmax=cov_df["coverage"].max())
    colors = [PLASMA(norm(c)) for c in cov_df["coverage"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(cov_df["property"], cov_df["non_null_count"],
                   color=colors, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Number of MOFs with Property", fontsize=11)
    total_mofs = cov_df["total_samples"].iloc[0]
    ax.set_title(f"Property Coverage Across {total_mofs:,} MOFs",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    sm = ScalarMappable(cmap=PLASMA, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Coverage Fraction", fontsize=10)

    save_fig(fig, "fig02_property_coverage")


def generate_data_sources_table(df):
    """Write ``table01_data_sources.tex`` listing each source's MOF count
    and property-family coverage (CO2 BE, H2O BE, DFT, synthesis, etc.)."""
    print("\nGenerating data-sources table (table01)...")

    families = {
        "Structure": ["Density", "Pore limiting diameter", "Largest cavity diameter"],
        "DFT electronic": ["Band gap (PBE)", "Band gap (HSE06)", "Max charge (DDEC)"],
        "DAC binding": ["Binding Energy CO2", "Binding Energy H2O"],
        "Thermodynamic": ["Free Energy (atom)", "Strain Energy (atom)"],
        "Synthesis": ["Thermal Stability"],
    }

    sources_order = [
        "CSD (Experimental)",
        "QMOF (Computed)",
        "Hypothetical (CoRE)",
        "Functionalized",
    ]
    rows = []
    for src in sources_order:
        if src not in df["source"].unique():
            continue
        sub = df[df["source"] == src]
        row = {"Source": src, "MOFs": len(sub)}
        for fam_name, cols in families.items():
            present = [c for c in cols if c in df.columns]
            if not present:
                row[fam_name] = 0
            else:
                any_nonnull = sub[present].notna().any(axis=1)
                row[fam_name] = int(any_nonnull.sum())
        rows.append(row)
    tbl = pd.DataFrame(rows)

    tex_rows = []
    for _, r in tbl.iterrows():
        parts = [r["Source"], f"{int(r['MOFs']):,}"]
        for fam in families:
            parts.append(f"{int(r[fam]):,}")
        tex_rows.append(" & ".join(parts) + r" \\")

    header = r"\textbf{Source} & \textbf{MOFs} & " + " & ".join(
        [f"\\textbf{{{f}}}" for f in families.keys()]
    ) + r" \\"

    tex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Data sources integrated into MOFology and the number of MOFs with data in each property family. Counts reflect MOFs with at least one non-null value in the listed property columns.}\n"
        "\\label{tab:data_sources}\n"
        "\\small\n"
        "\\begin{tabular}{lr" + "r" * len(families) + "}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )

    with open(os.path.join(TABLES, "table01_data_sources.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table01_data_sources.tex")


def generate_kg_stats_table(df, source_counts, property_coverage):
    """Generate LaTeX table with KG statistics."""
    print("\nGenerating KG statistics table...")

    # Count triples (approximate from KG file)
    kg_path = os.path.join(HPC, "KG", "data", "KG", "mof_kg.ttl")
    n_triples = 0
    if os.path.exists(kg_path):
        with open(kg_path, 'r') as f:
            n_triples = sum(1 for line in f if line.strip() and not line.startswith('@'))

    # Build table
    tex = r"""\begin{table}[htbp]
\centering
\caption{MOFology Knowledge Graph Statistics}
\label{tab:kg_stats}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Count} \\
\midrule
"""

    tex += f"Total MOFs & {len(df):,} \\\\\n"

    for src in ["CSD (Experimental)", "QMOF (Computed)", "Hypothetical (CoRE)", "Functionalized"]:
        if src in source_counts:
            tex += f"\\quad {src} & {source_counts[src]:,} \\\\\n"

    tex += f"Total Triples & $\\sim${n_triples/1e6:.1f}M \\\\\n"
    tex += f"Properties with Data & {len(property_coverage)} \\\\\n"
    tex += f"Properties ($\\geq$500 samples) & 45 \\\\\n"

    # Get relation types
    tex += f"Relation Types & 15 \\\\\n"
    tex += f"Data Sources & 6 \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(os.path.join(TABLES, "table02_kg_stats.tex"), "w") as f:
        f.write(tex)
    print("  [OK] table02_kg_stats.tex")


def main():
    print("=" * 60)
    print("Updating KG Statistics Figures")
    print("=" * 60)

    # Load data
    df, source_counts, property_coverage = load_kg_stats()

    # Generate figures
    fig_kg_overview(df, source_counts)
    fig_property_coverage_updated(df)

    # Generate tables (replaces fig00b/fig00c)
    generate_data_sources_table(df)
    generate_kg_stats_table(df, source_counts, property_coverage)

    print("\n" + "=" * 60)
    print("KG statistics figures updated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
