#!/bin/bash
# Run experiments with local paths
# Usage: ./run_local_experiments.sh

set -e

BASE="/home/matthart/Proj/MOFology/MOFKG_from_hpc"
STUDIES="$BASE/studies"

echo "======================================"
echo "MOFology Experiment Re-run (Local)"
echo "======================================"

# Step 1: Run property prediction (with family-aware splits - already implemented)
echo ""
echo "[1/4] Running property prediction with family-aware splits..."
cd "$STUDIES/src/ML_Chem"
python predict_chem.py 2>&1 | tee "$STUDIES/results/predict_chem_log.txt"

# Step 2: Run KG+Chem combo evaluation
echo ""
echo "[2/4] Running KG+Chem combo evaluation..."
cd "$STUDIES/src"
python run_chem_combo_eval.py \
    --chem_csv "$STUDIES/data/chemcial_properties.csv" \
    --kg_path "$BASE/KG/data/KG/mof_kg.ttl" \
    --compgcn_csv "$STUDIES/data/gnn_embeddings/mof_compgcn_embeddings_256d_3layers.csv" \
    --transe_csv "$STUDIES/data/transe_embeddings/mof_transe_embeddings_256d.csv" \
    --node2vec_pt "$STUDIES/data/node2vec/mof_embeddings_256d_p1.0_q1.0.pt" \
    --out_dir "$STUDIES/results/chem_combo_compare" \
    2>&1 | tee "$STUDIES/results/chem_combo_log.txt"

# Step 3: Run DAC screening
echo ""
echo "[3/4] Running DAC screening..."
python run_dac_screen.py \
    --kg_path "$BASE/KG/data/KG/mof_kg.ttl" \
    --chem_csv "$STUDIES/data/chemcial_properties.csv" \
    --pred_dir "$BASE/results/ML_Chem/prediction_results" \
    --compgcn_csv "$STUDIES/data/gnn_embeddings/mof_compgcn_embeddings_256d_3layers.csv" \
    --out_dir "$STUDIES/results/dac_screen" \
    2>&1 | tee "$STUDIES/results/dac_screen_log.txt"

# Step 4: Regenerate figures
echo ""
echo "[4/4] Regenerating paper figures..."
cd "$BASE/../paper/scripts"
python generate_figures.py 2>&1 | tee "$STUDIES/results/figure_gen_log.txt"

echo ""
echo "======================================"
echo "All experiments complete!"
echo "======================================"
echo "Results saved to: $STUDIES/results/"
echo "Figures saved to: $BASE/../paper/figures/generated/"
