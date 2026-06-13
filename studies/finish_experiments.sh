#!/bin/bash
# Finish experiments after background jobs complete
# Usage: ./finish_experiments.sh

set -e

BASE="/home/matthart/Proj/MOFology/MOFKG_from_hpc"
STUDIES="$BASE/studies"

echo "======================================"
echo "Waiting for background jobs to finish"
echo "======================================"
echo "Using 253K MOF dataset with family-aware splits"
echo ""

# Wait for predict_chem.py to finish
while pgrep -f "predict_chem.py" > /dev/null; do
    DONE=$(grep -c "Training Models for:" "$STUDIES/results/predict_chem_log.txt" 2>/dev/null || echo "0")
    TOTAL=$(grep -c "targets found" "$STUDIES/results/predict_chem_log.txt" 2>/dev/null | head -1 || echo "~46")
    echo "$(date '+%H:%M:%S') - Property prediction: $DONE targets processed..."
    sleep 60
done
echo "Property prediction complete!"

# Wait for chem_combo_eval to finish
while pgrep -f "run_chem_combo_eval.py" > /dev/null; do
    PROGRESS=$(tail -5 "$STUDIES/results/chem_combo_log.txt" 2>/dev/null | grep -oE "[0-9]+/[0-9]+" | tail -1 || echo "processing")
    echo "$(date '+%H:%M:%S') - Chem combo eval: $PROGRESS"
    sleep 30
done
echo "Chem combo evaluation complete!"

echo ""
echo "======================================"
echo "Running DAC Screening"
echo "======================================"

cd "$STUDIES/src"
python run_dac_screen.py \
    --kg_path "$BASE/KG/data/KG/mof_kg.ttl" \
    --chem_csv "$STUDIES/data/chemcial_properties.csv" \
    --pred_dir "$BASE/results/ML_Chem/prediction_results" \
    --compgcn_csv "$STUDIES/data/gnn_embeddings/mof_compgcn_embeddings_256d_3layers.csv" \
    --out_dir "$STUDIES/results/dac_screen" \
    2>&1 | tee "$STUDIES/results/dac_screen_log.txt"

echo ""
echo "======================================"
echo "Family-Aware Link Prediction Evaluation"
echo "======================================"

cd "$STUDIES/src"
python eval_link_prediction_family.py \
    --kg_path "$BASE/KG/data/KG/mof_kg.ttl" \
    --compgcn_csv "$STUDIES/data/gnn_embeddings/mof_compgcn_embeddings_256d_3layers.csv" \
    --transe_csv "$STUDIES/data/transe_embeddings/mof_transe_embeddings_256d.csv" \
    --out_dir "$STUDIES/results/link_prediction_family_eval" \
    2>&1 | tee "$STUDIES/results/link_pred_family_log.txt"

echo ""
echo "======================================"
echo "Regenerating Paper Figures"
echo "======================================"

cd "$BASE/../paper/scripts"
python generate_figures.py 2>&1 | tee "$STUDIES/results/final_figures_log.txt"

# Delete PDFs, keep only PNGs
rm -f "$BASE/../paper/figures/generated/"*.pdf
echo "Deleted PDF files (keeping PNGs only)"

echo ""
echo "======================================"
echo "All experiments complete!"
echo "======================================"
echo ""
echo "Results:"
echo "  - Property predictions: $BASE/results/ML_Chem/prediction_results/"
echo "  - Combo comparison: $STUDIES/results/chem_combo_compare/"
echo "  - DAC screening: $STUDIES/results/dac_screen/"
echo "  - Link prediction (family-aware): $STUDIES/results/link_prediction_family_eval/"
echo "  - Figures: $BASE/../paper/figures/generated/"
echo ""
echo "Check logs:"
echo "  - tail $STUDIES/results/predict_chem_log.txt"
echo "  - tail $STUDIES/results/chem_combo_log.txt"
echo "  - tail $STUDIES/results/dac_screen_log.txt"
echo "  - tail $STUDIES/results/link_pred_family_log.txt"
