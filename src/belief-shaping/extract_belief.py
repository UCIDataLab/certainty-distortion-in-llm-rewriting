import os
import sys
import random
import click
import pandas as pd
import tqdm

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jinja2 import Template
from utils import load_config_from_path
from model import load_model
from experiment_base import _sys, _user, extract_tag_content


def _load_api_key(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".txt"):
        return Path(path).read_text().strip()
    return path


@click.command()
@click.option("--claim_path", required=True, help="Path to input CSV file with extracted claims (output of extract_claim.py).")
@click.option("--claim_col", default="extracted_claim", show_default=True, help="Column name containing the extracted claims.")
@click.option("--belief_extract_prompt", required=True, help="Path to YAML with system_prompt, user_prompt, and answer_choices.")
@click.option("--belief_path", required=True, help="Path to output CSV with belief results.")
@click.option("--llm_judge_configs", required=True, help="Path to YAML/JSON config for the LLM judge.")
def extract_belief(claim_path, claim_col, belief_extract_prompt, belief_path, llm_judge_configs):
    if os.path.isfile(belief_path):
        print(f"Output already exists at {belief_path}. Skipping.")
        return

    df = pd.read_csv(claim_path)

    prompt_config = load_config_from_path(belief_extract_prompt)
    system_tmpl = Template(prompt_config["system_prompt"])
    user_tmpl = Template(prompt_config["user_prompt"])
    answer_choices = list(prompt_config["answer_choices"])

    judge_config = load_config_from_path(llm_judge_configs)
    conn_configs = judge_config["connection_configs"]
    conn_configs["api_key"] = _load_api_key(conn_configs.get("api_key", ""))
    judge = load_model(**judge_config["model_configs"], connection_configs=conn_configs)

    results = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting beliefs"):
        claim = row[claim_col]

        shuffled_choices = random.sample(answer_choices, len(answer_choices))
        system_prompt = system_tmpl.render(answer_choices=shuffled_choices)
        user_prompt = user_tmpl.render(claim=claim)

        response = judge.generate(messages=[_sys(system_prompt), _user(user_prompt)])
        example = row.to_dict()
        example.update({
            "belief_path": belief_path,
            claim_col: claim,
            "belief_answer": extract_tag_content(response, "answer"),
            "belief_score": extract_tag_content(response, "score"),
        })
        results.append(example)

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(belief_path)), exist_ok=True)
    out_df.to_csv(belief_path, index=False)
    print(f"Saved {len(out_df)} rows to {belief_path}")


if __name__ == "__main__":
    extract_belief()
