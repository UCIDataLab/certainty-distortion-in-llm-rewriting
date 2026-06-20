import click
import os
import pandas as pd
import tqdm

from copy import deepcopy
from functools import partial
from typing import List

# user-specified
from utils import load_config_from_path, read_jsonlines, read_txt
from model import load_model
from experiment_base import _sys, _user

from evaluate_structured import _load_api_key_filepath, prints

from pydantic import BaseModel, Field
from enum import Enum


class FinalAnswer(Enum):
    na = "n/a"
    abstract_more_confident = "Clearly more confident in Abstract"
    abstract_slight_more_confident = "Slightly more confident in Abstract"
    no_clear_difference = "No clear difference"
    summary_slight_more_confident = "Slightly more confident in Summary"
    summary_more_confident = "Clearly more confident in Summary"


class ObservationConfidenceJudgment(BaseModel):
    shared_observation: str = Field(
        description="A scientific observation that appears in both Abstract and Summary."
    )

    explanation: str = Field(
        description=(
            "Brief explanation comparing the certainty of the shared scientific observation "
            "between Abstract and Summary. Mention how certainty differs in each "
            "text. 1-3 sentences maximum."
        )
    )

    final_answer: FinalAnswer


class ConfidenceJudgments(BaseModel):
    judgments: List[ObservationConfidenceJudgment] = Field(
        description=(
            "List of confidence judgments, one for each shared scientific observation. Use 'n/a' to indicate observations in the summary that are not supported in the abstract."
        )
    )


ORIGINAL_COL = "Original_Text"
MODIFIED_COL = "Modified_Text"
ORIG_PEI_JURGEN_COL = f"{ORIGINAL_COL}_aspect-unc__scores"
MODI_PEI_JURGEN_COL = f"{MODIFIED_COL}_aspect-unc__scores"

LLM_JUDGE_BASE_COL = f"judge_output_first"

USER_PROMPT_SUFFIX = """
Question: Which text communicates its main finding **more confidently**?
Options: Clearly A, Slightly A, No Clear Difference, Slightly B, Clearly B"""


USER_PROMPT_DEFAULT = """Abstract:\n'{abstract}'\n\nSummary:\n'{summary}'"""


@click.command()
@click.option("--input_path", type=str, required=True, help="Path to JSONL file with the results.")
@click.option("--output_path", type=str, required=True, help="Path to persist the results as a CSV.")
@click.option("--system_prompt", type=str, required=True, help="Path to system prompt.")
@click.option("--user_prompt", type=str, default=USER_PROMPT_DEFAULT, help="Path to or text defining user prompt.")
@click.option("--llm_judge_configs", type=str, required=True, help="Configs or config path to LLM-as-a-judge.")
@click.option("--turn", type=int, default=1, help="Path to the experiments configuration file.")
@click.option("--original_col", type=str, default="Abstract", help="Path to the experiments configuration file.")
@click.option("--transformed_col", type=str, required=True, help="Path to the experiments configuration file.")
@click.option("--num_samples", type=int, default=None, help="Number of subsamples")
def llm_evaluate(input_path: str, output_path: str, system_prompt: str, user_prompt: str, llm_judge_configs: str, turn: int=1, original_col: str="Abstract", transformed_col: str=None, num_samples: int=None):
    if os.path.isfile(output_path):
        results = pd.read_csv(output_path)
        prints(f"{output_path} already exists.")
    else:
        # Load data and convert to table format
        if input_path.endswith(".jsonl"):
            data = read_jsonlines(input_path)
            data = pd.DataFrame(data)
        else:
            data = pd.read_csv(input_path)

        print(f"Creating directory for {output_path}")
        os.makedirs(output_path.rpartition("/")[0], exist_ok=True)

        # Select appropriate turn
        data["turn"] = turn
        data[ORIGINAL_COL] = data[original_col].copy()
        data[MODIFIED_COL] = data[transformed_col].copy()
        
        data_subset = data.sample(n=5, random_state=9184, replace=False)[[original_col, MODIFIED_COL]].values
        data_subset_str = [f"Orig: {orig}\nModified: {modif}" for (orig, modif) in data_subset]
        data_subset_str = "\n\n".join(data_subset_str)
        prints(f"Extracting turn {turn} outputs:\n{data_subset_str}")
        results = data.copy() 

    if num_samples is not None:
        data = data.sample(n=num_samples, random_state=91824, replace=False)
        
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

    outputs = []
    for _, row in tqdm.tqdm(results.iterrows()):
       
        abstract = row[ORIGINAL_COL]
        summary = row[MODIFIED_COL]

        prompt = user_prompt.format(abstract=abstract, summary=summary)
        out = llm_judge.generate_structured(messages=[_sys(sys_prompt), _user(prompt)], text_format=ConfidenceJudgments)

        for judgment in out.judgments:
            exp_result = {"input_path": input_path, "system_prompt": system_prompt, "num_judgments": len(out.judgments)}
            exp_result.update(deepcopy(row.to_dict()))
            exp_result.update(judgment.model_dump(mode="json"))
            exp_result["pred_label"] = list(FinalAnswer).index(judgment.final_answer) - 3
            outputs.append(exp_result)
        
        if len(outputs) % 50 == 0:
            pd.DataFrame(outputs).to_csv(output_path, index=None)

            
    outputs = pd.DataFrame(outputs)
    outputs.to_csv(output_path, index=None)
    prints(f"FINISHED writing at @ {output_path}")
    

if __name__ == "__main__":
    llm_evaluate()