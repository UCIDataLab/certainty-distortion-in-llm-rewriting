"""
Three prompting conditions for iterative paraphrasing experiments.

Assumptions:
- You already have `load_model(...)` available (from your snippet).
- The model client expects OpenAI-style chat messages:
  [{"role": "system"|"user"|"assistant", "content": "..."}]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import nltk
import re
import tqdm


# -----------------------------
# Utilities
# -----------------------------
def extract_tag_content(text: str, tag: str) -> str | None:
    """
    Extract content from XML-style tags in a string.
    
    Args:
        text: The string containing the tagged content
        tag: The tag name to extract content from
        
    Returns:
        The extracted content, or None if tag not found
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    return matches[-1].strip() if matches else text.strip()


def _sys(system_prompt: str) -> Dict[str, str]:
    return {"role": "system", "content": system_prompt}


def _user(text: str) -> Dict[str, str]:
    return {"role": "user", "content": text}


def _assistant(text: str) -> Dict[str, str]:
    return {"role": "assistant", "content": text}


def parse_numbered_bullets(text: str, expected_n: Optional[int] = None) -> List[str]:
    """
    Best-effort parser for outputs like:
      1) ...
      2. ...
      - 1. ...
      • 1) ...
    Returns a list of items in order.
    """
    import re

    text = re.sub(r"(Iteration )?([0-9]{1,3}).\s+", r"\g<2>. ", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Try "1. foo" / "1) foo" style parsing line-by-line
    items: List[str] = []
    pat = re.compile(r"^\s*(?:[-•]\s*)?(\d{1,4})\s*[\.\)]\s+(.*)\s*$")

    for ln in lines:
        m = pat.match(ln)
        if m:
            items.append(m.group(2).strip())

    # If that failed, try splitting on numbered markers inside text
    if not items:
        # Split on occurrences like "\n12. " or "\n12) "
        chunks = re.split(r"(?:^|\n)\s*(?:[-•]\s*)?\d{1,4}\s*[\.\)]\s+", text)
        chunks = [c.strip() for c in chunks if c.strip()]
        items = chunks

    if expected_n is not None and len(items) > expected_n:
        items = items[:expected_n]

    return items


# -----------------------------
# Base experiment
# -----------------------------
@dataclass
class RunResult:
    condition: str
    input_text: str
    outputs: List[str]  # paraphrases in order (turns)
    raw_outputs: List[str]  # raw model strings per turn
    transcripts: List[List[Dict[str, str]]]  # messages snapshot per turn


class Experiment:
    """
    Base class: defines common prompting pieces.
    You can override `build_messages_for_call(...)` to change behavior.
    """

    def __init__(
        self,
        model,
        system_prompt: str,
        user_preamble: str = None,
        user_postamble: str = None,
        n_turns: int = 100,
    ):
        self.model = model
        self.n_turns = n_turns
        self.system_prompt = system_prompt.format(n_turns=self.n_turns, n_sentences="{n_sentences}")

        self.user_preamble = user_preamble or ""
        if len(self.user_preamble):
            self.user_preamble = self.user_preamble.format(n_turns=self.n_turns)

        self.user_postamble = user_postamble or ""
        if len(self.user_postamble):
            self.user_postamble = self.user_postamble.format(n_turns=self.n_turns)

    def build_messages_for_call(
        self,
        *,
        original_text: str,
        current_text: str,
        history: List[str],
        n: int,
    ) -> List[Dict[str, str]]:
        """
        Default: single request with current_text.
        Subclasses should override.
        """
        user_prompt = (
            f"{self.user_preamble}"
            f"{current_text}"
            f"{self.user_postamble}"
        )
        
        # System prompt - update parameters if necessary
        n_sentences = len(nltk.sent_tokenize(original_text))
        system_prompt = self.system_prompt.format(n_sentences=n_sentences)
        # import pdb; pdb.set_trace()
        return [_sys(system_prompt), _user(user_prompt)]

    def run(self, text: str, n: int, **gen_kwargs) -> RunResult:
        raise NotImplementedError


# -----------------------------
# Condition 1: SingleShotExperiment
# -----------------------------
class SingleShotExperiment(Experiment):
    """
    One prompt, many requests in one response.

    This is useful when you want diversity but don't want iterative drift.
    """
    MAX_TOKENS_FIRST_REQUEST = 1000

    def run(self, text: str, **gen_kwargs) -> RunResult:
        # System prompt - update parameters if necessary
        n_sentences = len(nltk.sent_tokenize(text))
        system_prompt = self.system_prompt.format(n_sentences=n_sentences)
        # User prompt - update parameters if necessary
        user_prompt = (
            f"{self.user_preamble}" f"{text}" f"{self.user_postamble}"
        )
        # Create messages
        messages = [_sys(system_prompt), _user(user_prompt)]
        raw = self.model.generate(messages, 
                                  max_tokens=self.model.gen_kwargs["max_tokens"]-self.MAX_TOKENS_FIRST_REQUEST,
                                  **gen_kwargs)
        outputs = parse_numbered_bullets(raw, expected_n=self.n_turns)

        return RunResult(
            condition="SingleShotExperiment",
            input_text=text,
            outputs=outputs,
            raw_outputs=[raw],
            transcripts=[messages],
        )


# -----------------------------
# Condition 2: MultiNoHistoryExperiment
# -----------------------------
class MultiNoHistoryExperiment(Experiment):
    """
    Iterative paraphrasing where each call only sees the *latest paraphrase*,
    not the original and not the prior history.
    """

    def run(self, text: str, **gen_kwargs) -> RunResult:
        current = text
        outputs: List[str] = []
        raws: List[str] = []
        transcripts: List[List[Dict[str, str]]] = []

        for _step in tqdm.tqdm(range(self.n_turns), desc="No Turns:", disable=False):
            if len(text) == 0 or len(current) == 0:
                import pdb; pdb.set_trace()

            messages = self.build_messages_for_call(
                original_text=text,
                current_text=current,
                history=[],  # deliberately empty
                n=self.n_turns,
            )
            # import pdb; pdb.set_trace()
            raw = self.model.generate(messages, **gen_kwargs)
            
            # best-effort: take raw as the paraphrase (optionally strip)
            paraphrase = extract_tag_content(raw.strip(), tag="final_answer")
        
            # import pdb; pdb.set_trace()
            if len(paraphrase) == 0:
                paraphrase = "<EXTRACTION>"
                print("Could not extract final_answer.")

            outputs.append(paraphrase)
            raws.append(raw)
            transcripts.append(messages)

            current = paraphrase  # feed into next step

        return RunResult(
            condition="IndependentExperiment",
            input_text=text,
            outputs=outputs,
            raw_outputs=raws,
            transcripts=transcripts,
        )


# -----------------------------
# Condition 3: MultiHistoryExperiment
# -----------------------------
class MultiHistoryExperiment(Experiment):
    """
    Conversational flow:

      user: Text: <original>
      assistant: <paraphrase 1>
      user: Paraphrase the generated text
      assistant: <paraphrase 2>
      user: Paraphrase the generated text
      assistant: <paraphrase 3>
      ...

    Notes:
    - We keep the full chat history so the model "remembers" the original and prior paraphrases.
    - Each turn only asks to paraphrase the most recently generated text, as in your example.
    """
    def __init__(
        self,
        followup_user_prompt: str,
        discount_max_tokens: bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.followup_user_prompt = followup_user_prompt
        self.discount_max_tokens = int(discount_max_tokens)

    def run(self, text: str) -> RunResult:
        outputs: List[str] = []
        raws: List[str] = []
        transcripts: List[List[Dict[str, str]]] = []

        # System prompt - update parameters if necessary 
        n_sentences = len(nltk.sent_tokenize(text))
        system_prompt = self.system_prompt.format(n_sentences=n_sentences)
        
        # Start conversation with the original text as context.
        # We keep it in the chat so the model can always refer back.
        user_prompt = f"{self.user_preamble}{text}{self.user_postamble}".strip()
        messages: List[Dict[str, str]] = [
            _sys(system_prompt),
            _user(user_prompt),
        ]
        
        max_tokens = (self.model.gen_kwargs["max_tokens"] 
            - self.model.estimate_tokens(system_prompt) 
            - self.model.estimate_tokens(user_prompt)
        )
        raw, usage_tokens = self.model.generate(messages, 
                                                max_tokens=max_tokens,
                                                return_usage_tokens=True)
        paraphrase = raw.strip()

        # Update metadata / snapshot after each generation
        messages.append(_assistant(paraphrase))
        outputs.append(paraphrase)
        raws.append(raw)
        transcripts.append(list(messages))
        
        # Update max tokens
        max_tokens = self.model.gen_kwargs["max_tokens"] - usage_tokens - 10
        for _ in tqdm.tqdm(range(self.n_turns - 1), desc="No Turn", disable=False):        
            messages.append(_user(f"{self.followup_user_prompt}"))
            max_tokens -= self.model.estimate_tokens(f"{self.followup_user_prompt}", buffer=10)
            if max_tokens < 0:
                break
            raw, usage_tokens = self.model.generate(messages, max_tokens=max_tokens, return_usage_tokens=True)
            paraphrase = raw.strip()

            # Update metadata / snapshot after each generation
            messages.append(_assistant(paraphrase))
            outputs.append(paraphrase)
            raws.append(raw)
            transcripts.append(list(messages))
            
            # Update max tokens
            max_tokens = self.model.gen_kwargs["max_tokens"] - self.discount_max_tokens * usage_tokens

        return RunResult(
            condition="MultiHistoryExperiment",
            input_text=text,
            outputs=outputs,
            raw_outputs=raws,
            transcripts=transcripts,
        )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # You said this exists in your codebase:
    from model import load_model

    import os, json

    os.environ["CONNECTION_CONFIGS"] = json.dumps(
        {
            "api_key": "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"
        }
    )

    # Example:
    model = load_model(
        model_name="gpt-4.1-mini",
        model_device="openai",  # or "vllm" or "cuda:0"
        gen_kwargs={"temperature": 0.9, "max_tokens": 3000},
    )

    TEXT = "The quick brown fox jumps over the lazy dog."
    N = 5

    system_prompt = "You are a helpful assistant that produces faithful paraphrases."

    exp1 = SingleShotExperiment(
        model=model,
        system_prompt=system_prompt,
        user_preamble="Generate {n_turns} different paraphrases.",
        n_turns=N,
    )
    exp2 = MultiNoHistoryExperiment(model=model, system_prompt=system_prompt, n_turns=N)
    exp3 = MultiHistoryExperiment(
        model=model,
        system_prompt=system_prompt,
        n_turns=N,
        followup_user_prompt="Generate a different paraphrase. Return ONLY the paraphrased text.",
    )

    res1 = exp1.run(TEXT)
    print(res1.outputs[:3])

    res2 = exp2.run(TEXT)
    print(res2.outputs)

    res3 = exp3.run(TEXT)
    print(res3.outputs)
