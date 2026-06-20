BASE_DIR=./belief-shifting-experiments/yes-no/spiced
TASK=paraphrase-spiced-paper
# ## First call, let's determine the impact of original text (in spiced dataset this is "Paper_Finding")
# ## Because the original text is the same for paraphrasing and rewriting tasks (news), we can compute this once.
for EVALUATOR_MODEL in gemini-3.1-flash-lite; do # llama-70b qwen3-8b
   python src/belief-shaping/extract_belief_with_context.py \
      --claim_path $BASE_DIR/$TASK/extracted_claims.csv  \
      --claim_col extracted_claim \
      --context_col "Paper_Finding" \
      --belief_extract_context_prompt configs/experiments/belief-shaping/prompts/belief_extract_context_prompt.yaml \
      --belief_extract_context_path $BASE_DIR/paraphrase-spiced-paper/${EVALUATOR_MODEL}__belief_conditioned_on_original.csv \
      --llm_judge_configs configs/experiments/belief-shaping/models/$EVALUATOR_MODEL.yaml

   cp $BASE_DIR/paraphrase-spiced-paper $BASE_DIR/rewrite-spiced-news/${EVALUATOR_MODEL}__belief_conditioned_on_original.csv
done

## Second call, let's determine the impact of LLM-generated text (in spiced dataset this is "Modified_Finding")
# Mode 2 — Modified_Finding as context (jsonlines + turn):
for EVALUATOR_MODEL in gemini-3.1-flash-lite; do # # llama-70b qwen3-8b qwen3-vl-30b-a3b-instruct gpt-5-nano gpt-5.4; do
   for MODIFIER_MODEL in "gemini-3.1-flash-lite" "gpt-5-nano" "llama-v3p3-70b-instruct" "qwen3-8b"; #
   do
      echo "===================================================="
      echo "Processing beliefs for $MODIFIER_MODEL"
      echo "===================================================="
      python src/belief-shaping/extract_belief_with_context.py \
         --claim_path  $BASE_DIR/$TASK/extracted_claims.csv  \
         --input_path  ./top-p-sampling/$TASK/sample_0__${MODIFIER_MODEL}.jsonl \
         --target_turn 1 \
         --belief_extract_context_prompt configs/experiments/belief-shaping/prompts/belief_extract_context_prompt.yaml \
         --belief_extract_context_path $BASE_DIR/$TASK/${EVALUATOR_MODEL}__belief_conditioned_on_modified__$MODIFIER_MODEL-turn-1.csv  \
         --llm_judge_configs configs/experiments/belief-shaping/models/$EVALUATOR_MODEL.yaml
      echo "Done!"
   done
done