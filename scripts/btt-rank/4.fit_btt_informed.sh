#!/bin/bash
BASE=./output/uncertainty-ranking-pairwise

TASK="SPICED-news"
python src/rank/fit_btt.py \
    --input_path ${BASE}/data/${TASK}.csv \
    --input_path ${BASE}/data/${TASK}_informed.csv \
    --output_path ${BASE}/results-btt--${TASK}-informed.csv \
    --max_iter 300 \
    --tol 1e-4


TASK="MIMIC-CXR_combined_800"
python src/rank/fit_btt.py \
    --input_path ${BASE}/data/${TASK}.csv \
    --input_path ${BASE}/data/${TASK}_informed.csv \
    --output_path ${BASE}/results-btt--${TASK}-informed.csv \
    --max_iter 300 \
    --tol 1e-4
