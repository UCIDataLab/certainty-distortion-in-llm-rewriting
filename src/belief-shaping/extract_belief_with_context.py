import os
import sys
import random
import click
import pandas as pd
import tqdm

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jinja2 import Template
from utils import load_config_from_path, read_jsonlines
from model import load_model
from experiment_base import _sys, _user, extract_tag_content

CONTEXT_COL = "Modified_Finding"


def _load_api_key(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".txt"):
        return Path(path).read_text().strip()
    return path


@click.command()
@click.option("--claim_path", required=True, help="Path to CSV with extracted claims (output of extract_claim.py). Must contain 'Paper_Finding' and the claim column.")
@click.option("--context_col", default="Paper_Finding", show_default=True, help="Column in claim_path to use as context. Ignored when --input_path is provided.")
@click.option("--input_path", default=None, help="Path to JSONL results file with 'Paper_Finding' and 'exp__parsed_output'. When provided, extracts Modified_Finding at --target_turn and uses it as context.")
@click.option("--target_turn", default=None, type=int, help="Turn to extract from exp__parsed_output (1-indexed). Required when --input_path is provided.")
@click.option("--claim_col", default="extracted_claim", show_default=True, help="Column in claim_path containing the extracted claims.")
@click.option("--belief_extract_context_prompt", required=True, help="Path to YAML with system_prompt, user_prompt, and answer_choices.")
@click.option("--belief_extract_context_path", required=True, help="Path to output CSV with belief results.")
@click.option("--llm_judge_configs", required=True, help="Path to YAML/JSON config for the LLM judge.")
def extract_belief_with_context(claim_path, context_col, input_path, target_turn, claim_col, belief_extract_context_prompt, belief_extract_context_path, llm_judge_configs):
    if os.path.isfile(belief_extract_context_path):
        print(f"Output already exists at {belief_extract_context_path}. Skipping.")
        return

    if input_path and target_turn is None:
        raise click.UsageError("--target_turn is required when --input_path is provided.")

    # Load claims (Paper_Finding + extracted_claim)
    claims_df = pd.read_csv(claim_path)
    claims_df = claims_df.drop_duplicates(subset="Paper_Finding")

    if input_path:
        # Mode 2: extract Modified_Finding from jsonlines at the specified turn
        results_df = pd.DataFrame(read_jsonlines(input_path))
        valid_findings = set(claims_df["Paper_Finding"])
        results_df = results_df[results_df["Paper_Finding"].isin(valid_findings)].copy()
        results_df[CONTEXT_COL] = results_df["exp__parsed_output"].apply(lambda x: x[target_turn - 1])
        results_df["target_turn"] = target_turn

        df = claims_df.merge(results_df[["Paper_Finding", CONTEXT_COL, "target_turn"]], on="Paper_Finding", how="inner")
        n_dropped = len(claims_df) - len(df)
        if n_dropped:
            print(f"Warning: {n_dropped} rows from claim_path had no match in {input_path} and were dropped.")
        context_col = CONTEXT_COL
    else:
        # Mode 1: context column already present in claim_path
        df = claims_df.copy()
        df["target_turn"] = None

    prompt_config = load_config_from_path(belief_extract_context_prompt)
    system_tmpl = Template(prompt_config["system_prompt"])
    user_tmpl = Template(prompt_config["user_prompt"])
    answer_choices = list(prompt_config["answer_choices"])

    judge_config = load_config_from_path(llm_judge_configs)
    conn_configs = judge_config["connection_configs"]
    conn_configs["api_key"] = _load_api_key(conn_configs.get("api_key", ""))
    judge = load_model(**judge_config["model_configs"], connection_configs=conn_configs)

    results = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting beliefs with context"):
        claim = row[claim_col]
        context = row[context_col]

        shuffled_choices = random.sample(answer_choices, len(answer_choices))
        system_prompt = system_tmpl.render(answer_choices=shuffled_choices)
        user_prompt = user_tmpl.render(claim=claim, context=context)

        response = judge.generate(messages=[_sys(system_prompt), _user(user_prompt)])

        results.append({
            "belief_path": belief_extract_context_path,
            claim_col: claim,
            context_col: context,
            "target_turn": row.get("target_turn"),
            "belief_answer": extract_tag_content(response, "answer"),
            "belief_score": extract_tag_content(response, "score"),
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(belief_extract_context_path)), exist_ok=True)
    out_df.to_csv(belief_extract_context_path, index=False)
    print(f"Saved {len(out_df)} rows to {belief_extract_context_path}")


if __name__ == "__main__":
    extract_belief_with_context()
