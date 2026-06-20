import click
import os
import pandas as pd
import nltk
import re
import tqdm


from copy import deepcopy

# user-specified
from utils import load_config_from_path, read_jsonlines, read_txt
from model import load_model
from experiment_base import _sys, _user

from evaluate_structured import _load_api_key_filepath, prints, extract_turn

from pydantic import BaseModel, Field
from enum import Enum


class FinalAnswer(Enum):
    na = "n/a"
    broader_findings_text__more_confident = "Clearly more confident in Broader Findings Text"
    broader_findings_text__slight_more_confident = "Slightly more confident in Broader Findings Text"
    no_clear_difference = "No clear difference"
    shorter_text_slight_more_confident = "Slightly more confident in Shorter Text"
    shorter_text_more_confident = "Clearly more confident in Shorter Text"


class ConfidenceJudgment(BaseModel):
    shared_observation: str = Field(
        description="The shared radiological observation that appears in both Broader Findings Text and the Shorter Text."
    )

    explanation: str = Field(
        description=(
            "Brief explanation comparing the certainty of the shared radiological observation "
            "between Broader Findings Text and Shorter Text. Mention how certainty differs in each "
            "text. 1-3 sentences maximum."
        )
    )

    final_answer: FinalAnswer


ORIGINAL_COL = "Original_Text"
MODIFIED_COL = "Modified_Text"
ORIG_PEI_JURGEN_COL = f"{ORIGINAL_COL}_aspect-unc__scores"
MODI_PEI_JURGEN_COL = f"{MODIFIED_COL}_aspect-unc__scores"

LLM_JUDGE_BASE_COL = f"judge_output_first"

USER_PROMPT_DEFAULT = """Broader Findings Text:\n'{abstract}'\n\nShorter Text:\n'{summary}'"""


def extract_final_answer(text):
    # Strip <think>...</think> (closed or unclosed)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    
    # Prefer <final_answer> tag (last occurrence, closed or not)
    matches = list(re.finditer(r'<final_answer>', cleaned))
    if matches:
        remainder = cleaned[matches[-1].end():]
        close = re.search(r'</final_answer>', remainder)
        answer = remainder[:close.start()] if close else remainder
        return answer.strip(), 'tag'
    
    # Fallback: text after </think> in the original
    if '</think>' in text:
        return text.split('</think>', 1)[1].strip(), 'post_think'
    
    return text, 'original'



@click.command()
@click.option("--input_path", type=str, required=True, help="Path to JSONL file with the results.")
@click.option("--output_path", type=str, required=True, help="Path to persist the results as a CSV.")
@click.option("--system_prompt", type=str, required=True, help="Path to system prompt.")
@click.option("--user_prompt", type=str, default=USER_PROMPT_DEFAULT, help="Path to or text defining user prompt.")
@click.option("--llm_judge_configs", type=str, required=True, help="Configs or config path to LLM-as-a-judge.")
@click.option("--turn", type=int, default=1, help="Path to the experiments configuration file.")
@click.option("--original_col", type=str, default="FINDINGS", help="Path to the experiments configuration file.")
@click.option("--transformed_col", type=str, default="exp__parsed_output", help="Path to the experiments configuration file.")
@click.option("--num_samples", type=int, default=None, help="Number of subsamples")
@click.option("--uncertain_only", is_flag=True, default=False, help="If set, only filter uncertain samples")
def llm_evaluate(input_path: str, output_path: str, system_prompt: str, user_prompt: str, llm_judge_configs: str, turn: int=1, original_col: str="FINDINGS", transformed_col: str=None, num_samples: int=None, uncertain_only: bool = False):
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
        
        if transformed_col == "exp__parsed_output":
            data[MODIFIED_COL] = data[transformed_col].apply(extract_turn, turn=turn)
        else:
            data[MODIFIED_COL] = data[transformed_col].copy()

        data_subset = data.sample(n=5, random_state=9184, replace=False)[[original_col, MODIFIED_COL]].values
        data_subset_str = [f"Orig: {orig}\nModified: {modif}" for (orig, modif) in data_subset]
        data_subset_str = "\n\n".join(data_subset_str)
        prints(f"Extracting turn {turn} outputs:\n{data_subset_str}")
        results = data.copy() 

    # import pdb; pdb.set_trace()
    if num_samples is not None:
        results = results.sample(n=num_samples, random_state=91824, replace=False)

    if uncertain_only:
        results = results[results["_category"] == "uncertain"]
        print("Running eval for 'uncertain' subset:", len(results))
        
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
    for _, row in tqdm.tqdm(results.iterrows(), total=len(results)):
       
        broader_findings = row[ORIGINAL_COL]
        shorter_text_sentences, how_extr = extract_final_answer(row[MODIFIED_COL])

        if not shorter_text_sentences:
            print(f"Skipping example: {row[MODIFIED_COL]}")
            continue

        shorter_text_sentences = nltk.sent_tokenize(shorter_text_sentences)
        shorter_text_sentences = [s for s in shorter_text_sentences if len(s.strip()) > 5]
        
        for short_text in tqdm.tqdm(shorter_text_sentences, total=len(shorter_text_sentences)):
            prompt = user_prompt.format(abstract=broader_findings, summary=short_text)
            out = llm_judge.generate_structured(messages=[_sys(sys_prompt), _user(prompt)], text_format=ConfidenceJudgment)

            exp_result = {
                "input_path": input_path, 
                "system_prompt": system_prompt, 
                "content_extracted_post": how_extr,
            }
            exp_result.update(deepcopy(row.to_dict()))
            exp_result.update(out.model_dump(mode="json"))
            exp_result["pred_label"] = list(FinalAnswer).index(out.final_answer) - 3
            outputs.append(exp_result)

            if - 2 <= exp_result["pred_label"] < 0:
                print()
                prints(f"Underconfident example") 
                print(f"\n\n: Prompt: {prompt}\n\nOutput: {out.model_dump(mode='json')}")

            if 0 < exp_result["pred_label"] <= 2:
                print()
                prints(f"Overconfident example") 
                print(f"\n\n: Prompt: {prompt}\n\nOutput: {out.model_dump(mode='json')}")
            
        pd.DataFrame(outputs).to_csv(output_path, index=None)

            
    outputs = pd.DataFrame(outputs)
    outputs.to_csv(output_path, index=None)
    prints(f"FINISHED writing at @ {output_path}")
    

if __name__ == "__main__":
    llm_evaluate()