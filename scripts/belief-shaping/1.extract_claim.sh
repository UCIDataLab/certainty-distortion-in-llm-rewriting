## We will extract claims using a powerful LM

python src/belief-shaping/extract_claim.py \
   --input_path data/preprocessed/SPICED-news.csv \
   --target_col Paper_Finding \
   --claim_extract_prompt configs/experiments/belief-shaping/claim_extract_prompt.yaml \
   --claim_path ./belief-shaping-experiments/spiced/extracted_claims-news.csv \
   --llm_judge_configs configs/experiments/belief-shaping/gpt-5.4.yaml