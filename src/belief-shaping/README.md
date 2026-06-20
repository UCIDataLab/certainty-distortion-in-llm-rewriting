## Belief Shaping Experiments


The belief-shaping experiments can be divided into 4 main steps:

(1) **Extract claim from input column**: 
    Given an input file `--input_path`, a target column name (`--target_col`), and a claim extraction prompt (expressed as a jinja template) (`--claim_extract_prompt`), the goal is to iterate over each example in input_df[target_col] and
    use the `claim_extract_prompt`, replacing the `target_col` value in the jinja template with the value. The result should be appended to the original input file and stored in a column `extracted_claim` and then preserved in a file `--claim_path`. Note that if the file already exists,
    the pipeline should load it, and only if it doesn't exist it should call the LM.

(2) **Collect LM-judge belief for extracted claim**:
    Given the `--claim_path`, and a `--belief-extract-prompt` system prompt specified in a txt file using jinja template, collect the model's beliefs. The code should randomize the ordering of the answer options: `["Yes", "No", "Unknown"]`, 
    before replacing them in the belief-extract-prompt in the field `answer_choices`. 
    For each claim, we create a messages object, with the specified system prompt (with `answer_choies` replaced) and the user prompt (containing only the claim`). 
    Because the system prompt will instruct the model to generate a structured response, where the final answer and predicted scores will be enclosed in tags <answer></answer> and <score></score>, make sure to use `experiment_base.extract_tag_content` to extract those values. 
    The result should be stored in a new dataframe with columns [belief_path: str, belief_df[target_col], belief_answer, belief_score], which will be persisted in a specified `--belief_path` (also specified via CLI by the user).
    Similarly to before, this step is only conducted if the file does not exist already. So make sure to add a file exists check before running this step.

(3) **Collect LM-judge belief for extracted claim, under modified column** as context.
    Given the `--claim_path`,  a `--context_col`, and a `--belief_extract_context_prompt`. The experiment is similar to (2) but in addition to providing the claim in the user prompt we also want to provide the `context`, which is extracted from the `claim_path` file and column `context_col`.
    In other words, for each row in the file loaded from `claim_path`, select the row["claim"] and row[context_col] and provide them in the user prompt, which defaults to "Read this text carefully:\n'''{context}'''\n\nBased on the text do you agree with the claim '{claim}'?"
    The output should be stored in the `--belief_extract_context_path` as a CSV file. 

(4) **Comparison/analysis**: Given two filepaths `--belief1_path` and `--belief2_path`, compute the `belief shift rate (BSR)` and degree of agreement (DoA) metrics (i.e., the log-prob shift).

**Notes**:
- Assume that every input_path and output_path are CSV files.
- Assume that each part of the pipeline receives their own `llm_judge_configs`.
- Each part of the belief shaping experiment can be individual python script to keep it simple.
- Assume each prompt will be specified in a `.yaml` file containing two prompt definitions using jinja templates: `system_prompt` and `user_prompt`. There will be also another field called `answer_choices` which lists the choices to randomize before replacing them in the `system_prompt`.


**Disclaimer**: The code in this subfolder was generated w/ claude code and subsequently revised and checked by the authors of the paper.