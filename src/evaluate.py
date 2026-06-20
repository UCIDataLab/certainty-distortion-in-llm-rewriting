import click
import os
import pandas as pd
import tqdm

from functools import partial
from pathlib import Path
from typing import List

# user-specified
from utils import load_config_from_path, read_jsonlines, read_txt
from model import load_model
from experiment_base import _sys, _user

from uncertainty_utils import predict_pei_jurgens


ORIGINAL_COL = "Original_Text"
MODIFIED_COL = "Modified_Text"
ORIG_PEI_JURGEN_COL = f"{ORIGINAL_COL}_aspect-unc__scores"
MODI_PEI_JURGEN_COL = f"{MODIFIED_COL}_aspect-unc__scores"

LLM_JUDGE_BASE_COL = f"judge_output_first"

USER_PROMPT_SUFFIX = """
Question: Which text communicates its main finding **more confidently**?
Options:
- Clearly A
- Slightly A
- No Clear Difference
- Slightly B
- Clearly B"""



def _load_api_key_filepath(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".txt"):
        return Path(path).read_text()
    return path


def prints(msg, lim="-", qty=60):
    print(lim*qty)
    print(msg)
    print(lim*qty)
    print("\n\n")


def extract_turn(text: List[str], turn: int) -> str:
    return text[turn-1]


def bidirectional_eval(df: pd.DataFrame, llm_judge_generate: callable, system_prompt: str, colname1: str, colname2: str, print_freq: int=25) -> pd.DataFrame:
    prints(f"Running LLM-as-a-judge for {len(df)} and columns [{colname1}, {colname2}]")
    
    results = []
    for ix, example in tqdm.tqdm(df.iterrows()):
        example = example.to_dict()

        output1 = llm_judge_generate(messages=[_sys(system_prompt), _user(
            f"Text A: '{example[colname1].strip()}'\nText B: '{example[colname2].strip()}'\n{USER_PROMPT_SUFFIX}"
        )])            
        output2 = llm_judge_generate(messages=[_sys(system_prompt), _user(
            f"Text A: '{example[colname2].strip()}'\nText B: '{example[colname1].strip()}'\n{USER_PROMPT_SUFFIX}"
        )])    
        example[f"{LLM_JUDGE_BASE_COL}__{colname1}"] = output1
        example[f"{LLM_JUDGE_BASE_COL}__{colname2}"] = output2
        
        results.append(example)
        
        if len(results) % print_freq == 0:
            print("===")
            print(f"[Orig] Text A: {example[colname1].strip()}")
            print(f"[Modi] Text B: {example[colname2].strip()}")
            print(f"LLM Judge Preference ({colname1} first): {output1}")
            print(f"LLM Judge Preference ({colname2} first): {output2}")
    
    results = pd.DataFrame(results)
    return results


@click.command()
@click.option("--input_path", type=str, required=True, help="Path to JSONL file with the results.")
@click.option("--output_path", type=str, required=True, help="Path to persist the results as a CSV.")
@click.option("--system_prompt", type=str, required=True, help="Path to system prompt.")
@click.option("--llm_judge_configs", type=str, required=True, help="Configs or config path to LLM-as-a-judge.")
@click.option("--compute_other_certainty", type=bool, default=True, help="Whether to compute auxiliary certainty estimates.")
@click.option("--turn", type=int, default=1, help="Path to the experiments configuration file.")
@click.option("--original_col", type=str, default="Paper_Finding", help="Path to the experiments configuration file.")
@click.option("--transformed_col", type=str, default="exp__parsed_output", help="Path to the experiments configuration file.")
@click.option("--num_samples", type=int, default=None, help="Number of subsamples")
def llm_evaluate(input_path: str, output_path: str, system_prompt: str, llm_judge_configs: str, compute_other_certainty: bool = True, turn: int=1, original_col="Paper_Finding", transformed_col="exp__parsed_output", num_samples: int=None):
    if os.path.isfile(output_path):
        results = pd.read_csv(output_path)
        prints(f"{output_path} already exists.")
    else:
        # Load data and convert to table format
        data = read_jsonlines(input_path)
        data = pd.DataFrame(data)

        # Select appropriate turn
        data["turn"] = turn
        data[ORIGINAL_COL] = data[original_col].copy()
        data[MODIFIED_COL] = data[transformed_col].apply(extract_turn, turn=turn)

        data_subset = data.sample(n=5, random_state=9184, replace=False)[[original_col, MODIFIED_COL]].values
        data_subset_str = [f"Orig: {orig}\nModified: {modif}" for (orig, modif) in data_subset]
        data_subset_str = "\n\n".join(data_subset_str)
        prints(f"Extracting turn {turn} outputs:\n{data_subset_str}")
        results = data.copy() 

    if num_samples is not None:
        data = data.sample(n=num_samples, random_state=91824, replace=False)
        
    # Compute other certainty scores
    if compute_other_certainty and f"{MODI_PEI_JURGEN_COL}_mean" not in results.columns:
        results[f"{ORIG_PEI_JURGEN_COL}_mean"] = results[ORIGINAL_COL].apply(partial(predict_pei_jurgens, how="mean"))
        results[f"{MODI_PEI_JURGEN_COL}_mean"] = results[MODIFIED_COL].apply(partial(predict_pei_jurgens, how="mean"))

        results[f"{ORIG_PEI_JURGEN_COL}_last"] = results[ORIGINAL_COL].apply(partial(predict_pei_jurgens, how="last"))
        results[f"{MODI_PEI_JURGEN_COL}_last"] = results[MODIFIED_COL].apply(partial(predict_pei_jurgens, how="last"))
        
        prints(f"Persisting other certainty scores at:\n-> {output_path}")
        os.makedirs(output_path.rpartition("/")[0], exist_ok=True)
        results.to_csv(output_path, index=None)
    
    # Load llm-as-a-judge 
    if f"{LLM_JUDGE_BASE_COL}__{ORIGINAL_COL}" in results.columns:
        prints("Already computed LLM-as-a-judge. Skippping...")
        return 

    llm_judge_configs = load_config_from_path(llm_judge_configs)
    prints(f"Loading LLM-as-a-judge with configs: {llm_judge_configs}")

    conn_configs = llm_judge_configs.pop("connection_configs")
    conn_configs["api_key"] = _load_api_key_filepath(
        conn_configs.get("api_key", "")
    )
    llm_judge = load_model(**llm_judge_configs["model_configs"], connection_configs=conn_configs)

    # Load system prompt 
    sys_prompt = read_txt(system_prompt)
    prints(f"Loaded system prompt:\n{sys_prompt}")

    llm_judge_generate = partial(llm_judge.generate)
    results = bidirectional_eval(
        results,
        llm_judge_generate,
        sys_prompt,
        colname1=ORIGINAL_COL,
        colname2=MODIFIED_COL,
    )

    results.to_csv(output_path, index=None)
    

if __name__ == "__main__":
    llm_evaluate()