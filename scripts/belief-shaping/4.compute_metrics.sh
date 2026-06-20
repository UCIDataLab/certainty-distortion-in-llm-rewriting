
# Eval gemini on outputs of generator (gemini)
python src/belief-shaping/compute_metrics.py \
      --belief_path_a  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_original.csv \
      --belief_path_b  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_modified__gemini-3.1-flash-lite-turn-1.csv \
      --output_path debug_compute_metrics-gen-gemini.csv \
      --claim_col "extracted_claim" \
      --prior_belief_path "./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/baseline__gemini-3.1-flash-lite.csv"

python src/belief-shaping/compute_metrics.py \
      --belief_path_a  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_original.csv \
      --belief_path_b  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_modified__gpt-5-nano-turn-1.csv \
      --output_path debug_compute_metrics-gen-gpt.csv \
      --claim_col "extracted_claim" \
      --prior_belief_path "./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/baseline__gemini-3.1-flash-lite.csv"

python src/belief-shaping/compute_metrics.py \
      --belief_path_a  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_original.csv \
      --belief_path_b  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_modified__llama-v3p3-70b-instruct-turn-1.csv \
      --output_path debug_compute_metrics-gen-llama.csv \
      --claim_col "extracted_claim" \
      --prior_belief_path "./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/baseline__gemini-3.1-flash-lite.csv"

python src/belief-shaping/compute_metrics.py \
      --belief_path_a  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_original.csv \
      --belief_path_b  ./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/gemini-3.1-flash-lite__belief_conditioned_on_modified__qwen3-8b-turn-1.csv \
      --output_path debug_compute_metrics-gen-qwen.csv \
      --claim_col "extracted_claim" \
      --prior_belief_path "./belief-shifting-experiments/yes-no/spiced/paraphrase-spiced-paper/baseline__gemini-3.1-flash-lite.csv"


