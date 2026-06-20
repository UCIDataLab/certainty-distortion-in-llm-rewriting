from collections import defaultdict
from typing import Optional, List, Dict, Any

import json, jsonlines, os, wandb, warnings
import numpy as np
import pandas as pd


class Reporter:
    def add_example(self, *args, **kwargs):
        raise NotImplementedError

    def flush(self, path: str):
        raise NotImplementedError


# -----------------------------
# W&B logging loop
# -----------------------------
class WandbReporter(Reporter):
    """
    Buffers N examples. Each example is a list[dict] of length K (per-turn metrics).
    When flushed, logs a curve over k=1..K where values are averaged across examples.
    """

    def __init__(
        self,
        project: str,
        run_name: str,
        flush_every_n_examples: int = 10,
        tags: Optional[List[str]] = None,
        wandb_api_key: str = None,
        **kwargs,
    ):
        self.flush_every_n_examples = flush_every_n_examples
        self._examples: List[List[Dict[str, float]]] = []
        self._total_examples = 0

        if not wandb_api_key:
            wandb_api_key = os.environ["WANDB_API_KEY"]

        wandb.login(key=wandb_api_key)
        wandb.init(project=project, name=run_name, tags=tags or [], **kwargs)

        # Make x-axis explicit: x-axis is turn_k
        wandb.define_metric("*", step_metric="turn_k")

    def add_example(self, per_turn_metrics: List[Dict[str, float]], path: str):
        """
        per_turn_metrics: length K, each item is metric dict for that turn.
        """
        self._examples.append(per_turn_metrics)
        self._total_examples += 1

        if self._total_examples % self.flush_every_n_examples == 0:
            self.flush(path)

    def flush(self, path):
        if not self._examples:
            return

        with jsonlines.open(path, "w") as writer:
            writer.write_all(self._examples)

        print(len(self._examples))
        K = len(self._examples[0])
        # ^number of turns in each example
        # sanity: ensure all examples have same K
        for ex in self._examples:
            if len(ex) != K:
                warnings.warn(f"Inconsistent K: expected {K}, got {len(ex)}")

        # collect metrics per k across examples
        for k in range(K):
            # gather all keys at this turn (in case some dicts differ)
            keys = set()
            for ex in self._examples:
                keys.update(ex[k].keys())

            log_payload = {"turn_k": k + 1, f"examples_averaged": len(self._examples)}

            for key in keys:
                vals = []
                for ex in self._examples:
                    v = ex[k].get(key, None)
                    if v is None:
                        continue
                    vals.append(float(v))
                if vals:
                    log_payload[f"{key}_mean"] = float(np.mean(vals))

                    # Determine standard error
                    denom = max(1, np.sqrt(self._total_examples))
                    se = float(np.std(vals, ddof=1) / denom)
                    log_payload[f"{key}_lo"] = log_payload[f"{key}_mean"] - 1.96 * se
                    log_payload[f"{key}_hi"] = log_payload[f"{key}_mean"] + 1.96 * se

            wandb.log(log_payload)  # step comes from turn_k via define_metric

    def finish(self):
        wandb.finish()


class CSVReporter(Reporter):
    """
    Buffers N examples. Each example is a list[dict] of length K (per-turn metrics).
    When flushed, logs a curve over k=1..K where values are averaged across examples.
    """

    def __init__(
        self,
        flush_every_n_examples: int = 50,
    ):
        self.flush_every_n_examples = flush_every_n_examples
        self._examples: List[List[Dict[str, float]]] = []
        self._turns: List[int] = []
        self._k: int = -1
        self._metrics_names: List[str] = []
        self._total_examples = 0

    def add_example(self, per_turn_metrics: List[Dict[str, float]], path: str):
        """
        per_turn_metrics: length K, each item is metric dict for that turn.
        """
        self._examples.append(per_turn_metrics)
        self._turns.append(len(per_turn_metrics))
        self._k = max(self._k, len(per_turn_metrics))

        self._total_examples += 1
        for metric_res in per_turn_metrics:
            self._metrics_names.extend(list(metric_res.keys()))

        if self._total_examples % self.flush_every_n_examples == 0:
            self.flush(path)

    def _log_metric(self, metric_name, examples) -> pd.DataFrame:
        res = defaultdict(list)
        for i, example in enumerate(examples):
            for k in range(len(example)):
                res["metric_name"].append(metric_name)
                res["example"].append(i)
                res["turn"].append(k)
                res["metric_val"].append(example[k][metric_name])
        return pd.DataFrame(res)

    def flush(self, path: str):
        if not self._examples:
            return

        with jsonlines.open(path, "w") as writer:
            writer.write_all(self._examples)

        K = self._k
        # sanity: check if all examples have same K
        for ex in self._examples:
            if len(ex) != K:
                warnings.warn(f"Inconsistent K: expected {K}, got {len(ex)}")

        # collect metrics per k across examples
        for metric in self._metrics_names:
            res = self._log_metric(metric, examples=self._examples)
            metric_path = path.replace(".jsonl", f"_{metric}.csv")
            os.makedirs(metric_path.rpartition("/")[0], exist_ok=True)
            res.to_csv(metric_path, index=None)
