import os
import sys
import click
import pandas as pd
import tqdm

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jinja2 import Template
from utils import load_config_from_path
from model import load_model
from experiment_base import _sys, _user


def _load_api_key(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".txt"):
        return Path(path).read_text().strip()
    return path


@click.command()
@click.option("--input_path", required=True, help="Path to input CSV file.")
@click.option("--target_col", required=True, help="Column containing text to extract claims from.")
@click.option("--claim_extract_prompt", required=True, help="Path to YAML with system_prompt and user_prompt jinja templates.")
@click.option("--claim_path", required=True, help="Path to output CSV with extracted_claim column appended.")
@click.option("--llm_judge_configs", required=True, help="Path to YAML/JSON config for the LLM judge.")
def extract_claim(input_path, target_col, claim_extract_prompt, claim_path, llm_judge_configs):
    if os.path.isfile(claim_path):
        print(f"Output already exists at {claim_path}. Skipping.")
        return

    df = pd.read_csv(input_path)

    prompt_config = load_config_from_path(claim_extract_prompt)
    system_tmpl = Template(prompt_config["system_prompt"])
    user_tmpl = Template(prompt_config["user_prompt"])

    judge_config = load_config_from_path(llm_judge_configs)
    conn_configs = judge_config["connection_configs"]
    conn_configs["api_key"] = _load_api_key(conn_configs.get("api_key", ""))
    judge = load_model(**judge_config["model_configs"], connection_configs=conn_configs)

    results = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting claims"):
        text = row[target_col]
        ctx = {"text": text, target_col: text}
    
        messages = [
            _sys(system_tmpl.render(**ctx)),
            _user(user_tmpl.render(**ctx)),
        ]
        claim = judge.generate(messages=messages)

        record = row.to_dict()
        record["extracted_claim"] = claim.strip()
        results.append(record)

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(claim_path)), exist_ok=True)
    out_df.to_csv(claim_path, index=False)
    print(f"Saved {len(out_df)} rows to {claim_path}")


if __name__ == "__main__":
    extract_claim()
