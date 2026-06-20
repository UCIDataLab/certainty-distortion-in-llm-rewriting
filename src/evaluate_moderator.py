"""
Epistemic Certainty Judge with Position-Debiased Moderation
============================================================
Pipeline:
  1. Accept two pre-existing LLM verdicts (first_request, second_request) from
     runs where text order was swapped.
  2. Normalise both verdicts to the canonical [Text A, Text B] frame.
  3. If they agree  → return immediately (no extra API calls).
  4. If they conflict → run a dual-order, label-randomised moderator round.
       4a. Both moderator calls agree  → accept that verdict.
       4b. Moderator calls also conflict → forced Tie (genuinely borderline).
"""
import random
import click
import pandas as pd
import os, re, tqdm

# user-defined
import evaluate as eval

from enum import Enum
from functools import partial

# user-defined
from experiment_base import extract_tag_content

# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------
 
class Verdict(str, Enum):
    CLEARLY_A     = "Clearly A"
    SLIGHTLY_A    = "Slightly A"
    NO_CLEAR_DIFF = "No Clear Difference"
    SLIGHTLY_B    = "Slightly B"
    CLEARLY_B     = "Clearly B"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def extract_tag_content2(text: str, tag: str) -> str | None:
                """
                Extract content from XML-style tags in a string.

                Args:
                    text: The string containing the tagged content
                    tag: The tag name to extract content from

                Returns:
                    The extracted content, or None if tag not found
                """
                # First, try to find a properly closed tag (working backwards to avoid
                # matching tag names that appear as literal text inside other tags' content)
                parts = text.split(f"</{tag}>")
                if len(parts) > 1:
                    for part in reversed(parts[:-1]):
                        idx = part.rfind(f"<{tag}>")
                        if idx != -1:
                            return part[idx + len(f"<{tag}>"):].strip()

                # Fallback: unclosed tag — capture everything after the opening tag
                idx = text.rfind(f"<{tag}>")
                if idx != -1:
                    return text[idx + len(f"<{tag}>"):].strip()

                return text

def _extract_reasoning(raw: str) -> str:
    """Return everything before the <final_answer> tag as the reasoning text."""
    match = re.search(r"<final_answer>", raw, re.IGNORECASE)
    return raw[:match.start()].strip() if match else raw.strip()


def parse_verdict_from_tags(raw: str, label_a: str, label_b: str) -> Verdict:
    """
    Extract the decision from <final_answer> tags and remap neutral labels
    back to canonical Verdict values.
 
    label_a - the label that represents the canonical Text A in this call
    label_b - the label that represents the canonical Text B in this call
    """
    decision = extract_tag_content2(raw, 'final_answer')
    if len(decision) == len(raw):
        raise ValueError(f"No <final_answer> tags found in output:\n{raw!r}")

    d = decision.strip().lower()
    if d == f"clearly {label_a}".lower():
        return Verdict.CLEARLY_A
    elif d == f"slightly {label_a}".lower():
        return Verdict.SLIGHTLY_A
    elif d == "no clear difference":
        return Verdict.NO_CLEAR_DIFF
    elif d == f"slightly {label_b}".lower():
        return Verdict.SLIGHTLY_B
    elif d == f"clearly {label_b}".lower():
        return Verdict.CLEARLY_B
    else:
        raise ValueError(
            f"Unrecognised decision '{decision}' "
            f"(expected 'Clearly {label_a}', 'Slightly {label_a}', 'No Clear Difference', "
            f"'Slightly {label_b}', or 'Clearly {label_b}')"
        )
 
 
def normalize_verdict(raw: str, was_reversed: bool) -> Verdict:
    """
    Parse a first-round verdict and flip Text A <-> Text B when the texts
    were presented in reversed order.
    """
    v = parse_verdict_from_tags(raw, label_a="A", label_b="B")
    if was_reversed:
        flip = {
            Verdict.CLEARLY_A:     Verdict.CLEARLY_B,
            Verdict.SLIGHTLY_A:    Verdict.SLIGHTLY_B,
            Verdict.NO_CLEAR_DIFF: Verdict.NO_CLEAR_DIFF,
            Verdict.SLIGHTLY_B:    Verdict.SLIGHTLY_A,
            Verdict.CLEARLY_B:     Verdict.CLEARLY_A,
        }
        return flip.get(v, v)
    return v
 

# ---------------------------------------------------------------------------
# Moderator (single call)
# ---------------------------------------------------------------------------
def _moderate_single(
    moderator_generate: callable,
    moderator_prompt: str,
    *,
    text_first: str,
    text_second: str,
    label_first: str,
    label_second: str,
    first_verdict_str: str,
    second_verdict_str: str,
) -> tuple[Verdict, str]:
    """
    One moderator call. Returns (verdict, full_raw_output).
    label_first always maps to whichever canonical text was placed first.
    """
    prompt = moderator_prompt.format(
        label_first=label_first,
        label_second=label_second,
        first_verdict=first_verdict_str,
        second_verdict=second_verdict_str,
        text_first=text_first,
        text_second=text_second,
    )
    raw = moderator_generate(messages=[eval._user(prompt)])
    verdict = parse_verdict_from_tags(raw, label_a=label_first, label_b=label_second)
    return verdict, raw


# ---------------------------------------------------------------------------
# Dual-order, label-randomised moderation
# ---------------------------------------------------------------------------
def _moderate_with_debiasing(
    moderator_generate: callable,
    moderator_prompt: str,
    text_a: str,
    text_b: str,
    v1: Verdict,
    v2: Verdict,
    reasoning: dict,
) -> dict:
    """
    Run the moderator twice with swapped presentation order and randomised
    neutral labels to eliminate position and label-order bias.

    Mutates and returns the shared reasoning dict.
    """
    labels = random.sample(["Alpha", "Beta"], 2)
    label_for_a, label_for_b = labels

    print(f"Using labels: [{label_for_a}, {label_for_b}]")
    def to_neutral(v: Verdict) -> str:
        if v == Verdict.CLEARLY_A:     return f"Clearly {label_for_a}"
        if v == Verdict.SLIGHTLY_A:    return f"Slightly {label_for_a}"
        if v == Verdict.NO_CLEAR_DIFF: return "No Clear Difference"
        if v == Verdict.SLIGHTLY_B:    return f"Slightly {label_for_b}"
        if v == Verdict.CLEARLY_B:     return f"Clearly {label_for_b}"

    v1_str = to_neutral(v1)
    v2_str = to_neutral(v2)

    # Call 1: present [A, B]
    mod_v1, raw1 = _moderate_single(
        moderator_generate,
        moderator_prompt,
        text_first=text_a,
        text_second=text_b,
        label_first=label_for_a,
        label_second=label_for_b,
        first_verdict_str=v1_str,
        second_verdict_str=v2_str,
    )

    # Call 2: present [B, A]
    mod_v2_raw_label, raw2 = _moderate_single(
        moderator_generate,
        moderator_prompt,
        text_first=text_b,
        text_second=text_a,
        label_first=label_for_b,
        label_second=label_for_a,
        first_verdict_str=v1_str,
        second_verdict_str=v2_str,
    )

    # label_first in call 2 is label_for_b (= canonical Text B), so flip.
    flip = {
        Verdict.CLEARLY_A:     Verdict.CLEARLY_B,
        Verdict.SLIGHTLY_A:    Verdict.SLIGHTLY_B,
        Verdict.NO_CLEAR_DIFF: Verdict.NO_CLEAR_DIFF,
        Verdict.SLIGHTLY_B:    Verdict.SLIGHTLY_A,
        Verdict.CLEARLY_B:     Verdict.CLEARLY_A,
    }
    mod_v2 = flip[mod_v2_raw_label]

    reasoning["moderator_ab"] = _extract_reasoning(raw1)
    reasoning["moderator_ba"] = _extract_reasoning(raw2)

    if mod_v1 == mod_v2:
        return {
            "final_verdict": mod_v1,
            "method": "moderated_debiased_agreement",
            "reasoning": reasoning,
            label_for_a: text_a, 
            label_for_b: text_b,
        }
    else:
        reasoning["note"] = (
            "Moderator itself was inconsistent across orderings - "
            "case is genuinely too borderline to distinguish."
        )
        return {
            "final_verdict": Verdict.NO_CLEAR_DIFF,
            "method": "moderated_debiased_forced_tie",
            "reasoning": reasoning,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _empty_reasoning() -> dict:
    """Canonical reasoning dict with all fields set to None."""
    return {
        "judge_ab":     None,
        "judge_ba":     None,
        "moderator_ab": None,
        "moderator_ba": None,
        "note":         None,
    }
 

def evaluate(
    df,
    moderator_generate,
    moderator_prompt,
    text_a_col: str,
    text_b_col: str,
    first_request_ab_col: str,
    second_request_ba_col: str,
) -> dict:
    """
    Main entry point.
 
    Parameters
    ----------
    moderator_generate  : moderator generator callable
    moderator_prompt    : moderator template prompt
    text_a_col          : canonical Text A (as shown in first_request)
    text_b_col          : canonical Text B (as shown in first_request)
    first_request_ab_col   : raw judge output when order was [A, B]
    second_request_ba_col  : raw judge output when order was [B, A]
 
    Returns
    -------
    dict with keys:
        final_verdict  - Verdict enum value
        method         - how the verdict was reached
        reasoning      - dict with keys:
                           judge_ab, judge_ba, moderator_ab, moderator_ba, note
                         (None for stages that were not reached)
    """
    results = []
    for _, example in tqdm.tqdm(df.iterrows(), total=len(df)):
        example = example.to_dict()

        text_a = example[text_a_col]
        text_b = example[text_b_col]

        first_request = example[first_request_ab_col]
        second_request = example[second_request_ba_col]

        reasoning = _empty_reasoning()
        reasoning["judge_ab"] = _extract_reasoning(first_request)
        reasoning["judge_ba"] = _extract_reasoning(second_request)
    

        # first_request = extract_tag_content2(first_request, 'final_answer')
        # second_request = extract_tag_content2(second_request, 'final_answer')
        v1 = normalize_verdict(first_request, was_reversed=False)
        v2 = normalize_verdict(second_request, was_reversed=True)
    
        if v1 == v2:
            output = {
                "final_verdict": v1,
                "method": "first_round_agreement",
                "reasoning": reasoning,
            }
        else:
            print("Debiasing using judge")
            output = _moderate_with_debiasing(moderator_generate, moderator_prompt, text_a, text_b, v1, v2, reasoning)
        
        example["moderator_verdict"] = output
        results.append(example)
        
    results = pd.DataFrame(results)
    return results


@click.command()
@click.option("--input_path", type=str, required=True, help="Path to CSV file with the results of first round of eval.")
@click.option("--output_path", type=str, required=True, help="Path to persist the results as a CSV.")
@click.option("--system_prompt", type=str, required=True, help="Path to moderation prompt.")
@click.option("--llm_judge_configs", type=str, required=True, help="Configs or config path to LLM-as-a-judge.")
# @click.option("--input_path", type=str, default="/extra/ucinlp1/cbelem/projects/verbal-unc-propagation/uncertainty-in-llms-evals/rewrite-spiced-news/prompt_sys-reasoning-5options-structured/greedy-decoding/sample_0__gemini-3.1-flash-lite__turn-1.csv")
# @click.option("--output_path", type=str, default="debug.csv")
# @click.option("--system_prompt", type=str, default="/home/cbelem/projects/uncertainty-in-llms/configs/experiments/evaluation/prompts/moderator.txt")
# @click.option("--llm_judge_configs", type=str, default="/home/cbelem/projects/uncertainty-in-llms/configs/experiments/belief-shift/models/gpt-5.4-mini.yaml")
def llm_evaluate_with_moderator(input_path: str, output_path: str, system_prompt: str, llm_judge_configs: str):
    base_output_dir = output_path.rpartition("/")[0]
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Load data and convert to table format
    results = pd.read_csv(input_path)

    llm_judge_configs = eval.load_config_from_path(llm_judge_configs)
    eval.prints(f"Loading LLM-as-a-judge with configs: {llm_judge_configs}")

    conn_configs = llm_judge_configs.pop("connection_configs")
    conn_configs["api_key"] = eval._load_api_key_filepath(conn_configs.get("api_key", ""))
    llm_judge = eval.load_model(**llm_judge_configs["model_configs"], connection_configs=conn_configs)

    # Load system prompt 
    sys_prompt = eval.read_txt(system_prompt)
    eval.prints(f"Loaded system prompt:\n{sys_prompt}")

    llm_judge_generate = partial(llm_judge.generate)
    random.seed(18274)
    results = evaluate(
        results,
        moderator_generate=llm_judge_generate,
        moderator_prompt=sys_prompt,
        text_a_col = eval.ORIGINAL_COL,
        text_b_col = eval.MODIFIED_COL,
        first_request_ab_col=f"{eval.LLM_JUDGE_BASE_COL}__{eval.ORIGINAL_COL}",
        second_request_ba_col=f"{eval.LLM_JUDGE_BASE_COL}__{eval.MODIFIED_COL}",
    )
    try:
        results["llm_configs"] = llm_judge_configs
    except:
        pass
    
    results.to_csv(output_path, index=None)


if __name__ == "__main__":
    llm_evaluate_with_moderator()