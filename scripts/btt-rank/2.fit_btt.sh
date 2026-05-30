python src/rank/fit_btt.py \
    --input_path ./output/uncertainty-ranking-pairwise/data/SPICED-news.csv \
    --output_path ./output/uncertainty-ranking-pairwise/results-btt--SPICED-news.csv \
    --max_iter 300 \
    --tol 1e-4

python src/rank/fit_btt.py \
    --input_path ./output/uncertainty-ranking-pairwise/data/MIMIC-CXR_combined_800.csv \
    --output_path ./output/uncertainty-ranking-pairwise/results-btt--MIMIC-CXR_combined_800.csv \
    --max_iter 300 \
    --tol 1e-4