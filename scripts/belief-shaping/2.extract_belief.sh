BASE_DIR=./belief-shifting-experiments/yes-no/spiced
### PARAPHRASE TASK
TASK=paraphrase-spiced-paper

for MODEL in llama-70b qwen3-8b; do
   python src/belief-shaping/extract_belief.py \
      --claim_path $BASE_DIR/$TASK/extracted_claims.csv \
      --claim_col extracted_claim \
      --belief_extract_prompt configs/experiments/belief-shaping/prompts/belief_extract_prompt.yaml \
      --belief_path $BASE_DIR/$TASK/baseline__$MODEL.csv \
      --llm_judge_configs configs/experiments/belief-shaping/models/$MODEL.yaml
done

