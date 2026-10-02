#!/bin/bash
# Convenience script: runs the M1-M5 progression, then (optionally) the three
# ablation-isolation variants. Comment out lines you don't have time for.
# Run from inside the retrain/ directory.
set -e

DATA_ROOT="../dataset/SKIN_CLEAN_DEDUP"
RESULTS="../results"

# --- priority: the M1-M5 progression (mirrors the paper's main narrative) ---
python train.py --stage m1 --data_root $DATA_ROOT --out_dir $RESULTS/m1 --epochs 20 --patience 5
python train.py --stage m2 --data_root $DATA_ROOT --out_dir $RESULTS/m2 --epochs 20 --patience 5
python train.py --stage m3 --data_root $DATA_ROOT --out_dir $RESULTS/m3 --epochs 20 --patience 5
python train.py --stage m4 --data_root $DATA_ROOT --out_dir $RESULTS/m4 --epochs 25 --patience 5
python train.py --stage m5 --data_root $DATA_ROOT --out_dir $RESULTS/m5 --epochs 15 --patience 5

# --- bonus (only if time permits before the deadline): isolates each M5 component ---
# python train.py --stage m5_caw_only      --data_root $DATA_ROOT --out_dir $RESULTS/m5_caw_only      --epochs 15 --patience 5
# python train.py --stage m5_contrast_only --data_root $DATA_ROOT --out_dir $RESULTS/m5_contrast_only --epochs 15 --patience 5
# python train.py --stage m5_gabor_only    --data_root $DATA_ROOT --out_dir $RESULTS/m5_gabor_only    --epochs 15 --patience 5

echo "All requested runs complete. Results are in $RESULTS/<stage>/metrics.json"
