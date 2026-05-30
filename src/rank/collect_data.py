"""
Rao-Kupper Model — Bradley-Terry with Ties
===========================================
**Obtain the preference data with ties**. For the input data,
we extract the column `text`, and for each text_i we perform
k comparisons with text_j != text_i. Comparisons will be determined
using an llm-as-a-judge approach ($text_i$, $text_j$) $\in$ {i, j, tie}.
This function returns one of three labels "i" if text_i wins over text_j,
"j" if text_j wins over text_i, and tie if neither wins.
We will record a matrix of all wins, and a matrix of all Ties.
Each matrix will be of shape len(texts) x len(texts) and diagonal
columns will be pre-filled with 0.
"""

import click
import os
import random
import pandas as pd
import numpy as np
import tqdm

import sys
sys.path.append("..")

from pathlib import Path
from typing import List, Optional, Tuple

from utils import load_config_from_path, read_txt
from model import load_model
from evaluate import _load_api_key_filepath, prints
from experiment_base import _sys, _user, extract_tag_content


def parse_judge_label(raw_output: str) -> str:
    """Extract 'A', 'B', or 'tie' from the judge's raw output."""
    label = extract_tag_content(raw_output, tag="final_answer").strip().lower()
    if label == "text a":
        return "A"
    if label == "text b":
        return "B"
    if "tie" in label:
        return "Tie"
    # fallback: scan for first occurrence of a valid label
    for token in label.split():
        if token in ("text a", "text b", "tie"):
            return token.upper()
    else:
        import pdb; pdb.set_trace()
        # unexpected
    return "tie"


def collect_votes(
    llm_judge,
    system_prompt: str,
    text_i: str,
    text_j: str,
    n_votes: int,
) -> Tuple[int, int, int]:
    """
    Run a bidirectional LLM judge with generate_multiple.

    Calls the judge twice per pair (swapping A/B order to debias position bias),
    each time requesting n_votes completions.

    Returns (wins_i, wins_j, ties) aggregated over both orderings.
    """
    wins_i, wins_j, ties = 0, 0, 0

    # Ordering 1: text_i = A, text_j = B
    messages1 = [_sys(system_prompt), _user(
        f"Text A: '{text_i.strip()}'\nText B: '{text_j.strip()}'"
    )]
    for raw in llm_judge.generate_multiple(messages1, n=n_votes):
        label = parse_judge_label(raw)
        if label == "A":
            wins_i += 1
        elif label == "B":
            wins_j += 1
        else:
            ties += 1

    # Ordering 2: text_j = A, text_i = B  (swapped to debias)
    messages2 = [_sys(system_prompt), _user(
        f"Text A: '{text_j.strip()}'\nText B: '{text_i.strip()}'"
    )]
    for raw in llm_judge.generate_multiple(messages2, n=n_votes):
        label = parse_judge_label(raw)
        if label == "A":
            wins_j += 1   # j was A in this ordering
        elif label == "B":
            wins_i += 1   # i was B in this ordering
        else:
            ties += 1

    return wins_i, wins_j, ties


@click.command()
@click.option("--input_path", type=str, required=True, help="Path to CSV file with the raw data.")
@click.option("--output_path", type=str, required=True, help="Path to persist the results as a CSV.")
@click.option("--system_prompt", type=str, required=True, help="Path to the system prompt .txt file.")
@click.option("--llm_judge_configs", type=str, required=True, help="Path to the LLM-as-a-judge config file.")
@click.option("--k", type=int, default=5, help="Number of random texts from paper_col to compare each original text against.")
@click.option("--n_votes", type=int, default=1, help="Number of generate_multiple responses per pair per ordering.")
@click.option("--paper_col", type=str, default="Paper_Finding", help="Column with the candidate pool to sample comparisons from.")
@click.option("--num_samples", type=int, default=None, help="If set, subsample this many rows from the dataset.")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
@click.option("--strength_estimate_filepath", type=str, default=None, help="Path to BTT results CSV (with 'strength_btt' and 'text' columns). If set, restricts eligible candidates to the K*2 texts with closest strength to each text_i.")
def collect_preference_data(
    input_path: str,
    output_path: str,
    system_prompt: str,
    llm_judge_configs: str,
    k: int,
    n_votes: int,
    paper_col: str,
    num_samples: Optional[int],
    seed: int,
    strength_estimate_filepath: Optional[str],
):
    random.seed(seed)
    np.random.seed(seed)

    if os.path.isfile(output_path):
        prints(f"{output_path} already exists. Loading and skipping data collection.")
        return

    # Load data
    data = pd.read_csv(input_path)

    if num_samples is not None:
        data = data.sample(n=num_samples, random_state=seed, replace=False).reset_index(drop=True)

    originals: List[str] = data[paper_col].tolist()
    n_orig = len(originals)
    prints(f"Loaded {n_orig} original texts.\nRunning {k} comparisons per original with {n_votes} vote(s) per ordering.")

    # Load LLM judge
    judge_configs = load_config_from_path(llm_judge_configs)
    prints(f"Loading LLM-as-a-judge with configs: {judge_configs}")
    conn_configs = judge_configs.pop("connection_configs")
    conn_configs["api_key"] = _load_api_key_filepath(conn_configs.get("api_key", ""))
    llm_judge = load_model(**judge_configs["model_configs"], connection_configs=conn_configs)

    # Load system prompt
    sys_prompt = read_txt(system_prompt)
    prints(f"Loaded system prompt:\n{sys_prompt}")

    # Load strength estimates for strength-guided sampling
    text_to_strength = None
    if strength_estimate_filepath is not None:
        strength_df = pd.read_csv(strength_estimate_filepath)
        text_to_strength = dict(zip(strength_df["text"], strength_df["strength_btt"]))
        prints(f"Loaded strength estimates for {len(text_to_strength)} texts from {strength_estimate_filepath}.")

    # Collect preference data
    results = []
    for i, text_i in enumerate(tqdm.tqdm(originals, desc="Originals")):
        eligible = [j for j in range(n_orig) if j != i]

        if text_to_strength is not None:
            strength_i = text_to_strength.get(text_i)
            if strength_i is not None:
                # Restrict to K*2 candidates closest in strength; unknown texts sort last
                pool_size = k * 2
                eligible = sorted(
                    eligible,
                    key=lambda j: abs(text_to_strength.get(originals[j], float("inf")) - strength_i),
                )
                eligible = eligible[:pool_size]

        sampled_js = random.sample(eligible, min(k, len(eligible)))

        for j in sampled_js:
            text_j = originals[j]
            wins_i, wins_j, ties = collect_votes(llm_judge, sys_prompt, text_i, text_j, n_votes)

            results.append({
                "idx_i": i,
                "idx_j": j,
                "text_i": text_i,
                "text_j": text_j,
                "wins_i": wins_i,
                "wins_j": wins_j,
                "ties": ties,
                "total_votes": wins_i + wins_j + ties,
            })

    results_df = pd.DataFrame(results)
    out_dir = output_path.rpartition("/")[0]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(output_path, index=None)
    prints(f"Saved {len(results_df)} comparisons to:\n-> {output_path}")


if __name__ == "__main__":
    collect_preference_data()
