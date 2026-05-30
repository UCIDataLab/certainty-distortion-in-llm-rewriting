#!/bin/bash
# TASK="SPICED-news"

BASE_DIR=.
python src/rank/collect_data.py \
    --input_path  data/preprocessed/$TASK.csv \
    --output_path $BASE_DIR/output/uncertainty-ranking-pairwise/data/${TASK}_informed.csv \
    --system_prompt configs/experiments/evaluation/sys-reasoning.txt \
    --llm_judge_configs configs/experiments/evaluation/judge-gpt-5.4-mini-temp0.7.yaml \
    --k 5 --n_votes 3 --strength_estimate_filepath "$BASE_DIR/output/uncertainty-ranking-pairwise/results-btt--SPICED-news.csv"

# Advice - How to obtain better rankings
# - First stage, need to run k > log(N=397) ~ 2.60 to ensure that a fully connected graph through
# random walks. Estimate strengths; Nu will be high due to uninformativeness of ties.
# To obtain that data, you should first run: 1.collect_data.sh and then 2.fit_btt.sh
# - Second stage, to maximize the Fisher information, we will be able to obtain more information
# by comparing neighboring results, we obtain more useful information to further improve the ranking.
# Estimate strengths in the concatenated data from first stage + second stage. 


TASK=MIMIC-CXR_combined_800

python src/rank/collect_data.py \
    --input_path  data/preprocessed/$TASK.csv \
    --output_path $BASE_DIR/output/uncertainty-ranking-pairwise/data/${TASK}_informed.csv \
    --system_prompt configs/experiments/evaluation/prompts/sys-reasoning.txt \
    --llm_judge_configs configs/experiments/evaluation/judge-gpt-5.4-mini-temp0.7.yaml \
    --strength_estimate_filepath "$BASE_DIR/output/uncertainty-ranking-pairwise/results-btt--$TASK.csv" \
    --k 8 --n_votes 3 --paper_col sentence 
