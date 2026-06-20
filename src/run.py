""" """

from pathlib import Path
from typing import List

import click
import copy
import os, json
import nltk
import pandas as pd
import tqdm

import experiment_base as exp
import model as m
import utils as ut
import data as dt

BASE_DIR = "/home/cbelem/projects/uncertainty-in-llms"
FORMATTED_INPUT_COL = "prompt"

def _load_api_key_filepath(path: str) -> str:
    if os.path.isfile(path) and path.endswith(".txt"):
        return Path(path).read_text()
    return path


def run_experiment_with_reporting(
    data: pd.DataFrame,
    experiment: exp.Experiment,
    output_path: str,
):
    data = data.copy()
    
    counts = 0
    prefix = ""
    with open(output_path, "w") as f:
        for _, orig_example in tqdm.tqdm(data.iterrows(), total=len(data), desc="No Examples"):
            uuid = ut.assign_uuid(orig_example.to_dict())
            orig_example["uuid"] = uuid
            orig_example["input_num_sents"] = len(
                nltk.sent_tokenize(orig_example[dt.FORMATTED_INPUT_COL])
            )
            orig_example["input_num_words"] = len(
                nltk.word_tokenize(orig_example[dt.FORMATTED_INPUT_COL].strip())
            )
            example = copy.deepcopy(orig_example)
            run_res = experiment.run(text=example[dt.FORMATTED_INPUT_COL])

            example["exp__condition"] = run_res.condition
            example["exp__input_text"] = run_res.input_text
            example["exp__parsed_output"] = run_res.outputs
            example["exp__raw_output"] = run_res.raw_outputs
            example["exp__all_transcripts"] = run_res.transcripts[-1]

            f.write(prefix + json.dumps(example.to_dict(), sort_keys=True))
            prefix = "\n"

            if counts % 50 == 0:
                print(f"-------------------------------- Example {counts} --------------------------------")
                print(f"Example: {example[dt.FORMATTED_INPUT_COL]}")
                print(f"Output: {run_res.outputs}")
                print(f"Condition: {run_res.condition}")
                print(f"Input Text: {run_res.input_text}")
                if len(run_res.outputs) > 10:
                    print(f"Parsed Output: \n- {"\n- ".join(run_res.outputs[:5] + run_res.outputs[-5:])}")
                else:
                    print(f"Parsed Output: \n- {"\n- ".join(run_res.outputs)}")
                print("-" * 80)
            counts += 1

@click.command()
@click.option("--experiments_config", type=str, required=True, help="Path to the experiments configuration file.")
def generate(experiments_config: str):
    configs = ut.load_config_from_path(experiments_config)
    # -------------------- --------------------
    # Load Model
    # -------------------- --------------------
    connection_configs = configs["connection_configs"]
    connection_configs["api_key"] = _load_api_key_filepath(
        connection_configs.get("api_key", "")
    )

    model_configs = configs["model_configs"]
    model = m.load_model(**model_configs, connection_configs=connection_configs)

    # -------------------- --------------------
    # Load data
    # -------------------- --------------------
    data = dt.load_dataset(**configs["data_configs"])
    # -------------------- --------------------
    # Run generation
    # -------------------- --------------------
    n_samples = configs["experiment_configs"].pop("n_samples")
    output_path = configs["experiment_configs"].pop("output_path")
    os.makedirs(output_path.rpartition("/")[0], exist_ok=True)

    for i in range(n_samples):
        dir, _, basename = output_path.rpartition("/")
        print(f"Creating directory for {dir}")
        os.makedirs(dir, exist_ok=True)
        sample_output_path = f"{dir}/sample_{i}__{basename}"
        print(f"Results will be persisted @ {sample_output_path}")

        # -------------------- --------------------
        # Load Experiment Object (from configs)
        # -------------------- --------------------
        experiment_obj = ut.load_class_from_config(
            configs["experiment_configs"], model=model
        )
        run_experiment_with_reporting(
            data,
            experiment_obj,
            output_path=sample_output_path
        )


if __name__ == "__main__":
    generate()
