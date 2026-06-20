"""
Goal:
- Run an experiment that generates paraphrases over multiple turns.
- After every K completed runs, compute metrics using a `clf` by comparing each paraphrase
  against the ORIGINAL text.
- Average metrics across ALL paraphrase turns (and across those K runs), then log to W&B.

Assumptions about `clf`:
- It exposes: clf.score(original: str, paraphrase: str) -> Dict[str, float]
  Example return: {"semantic_sim": 0.87, "toxicity": 0.02}
You can adapt `score_pair(...)` below to your actual clf interface.

W&B:
- pip install wandb
- wandb.init(...) done in main
"""

from __future__ import annotations
from collections import Counter
from functools import cache

# Length metrics
from nltk import sent_tokenize, word_tokenize

# Similarity metrics computation
from rapidfuzz import distance, utils
from sentence_transformers import SentenceTransformer
from minicheck.minicheck import MiniCheck

# Uncertainty metric computation
from certainty_estimator.predict_certainty import CertaintyEstimator
from scipy.stats import entropy

from typing import Dict, List, Any, Optional
from experiment_base import RunResult
import numpy as np


def stable_softmax(x, axis=1):
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    softmax = numerator / denominator
    return softmax


# Length-related scoring
@cache
def score_length(original: str, paraphrase: str, **kwargs) -> Dict[str, float]:
    """Compute the difference in length between original and paraphrased version."""
    orig_wlen = len(word_tokenize(original.strip()))
    para_wlen = len(word_tokenize(paraphrase.strip()))

    orig_slen = len(sent_tokenize(original.strip()))
    para_slen = len(sent_tokenize(paraphrase.strip()))

    return {
        "length/num_words_diff": para_wlen - orig_wlen,
        "length/num_sents_diff": para_slen - orig_slen,
    }


# Score diversity scoring
@cache
def score_diversity(
    original: str,
    paraphrase: str,
    emb_model: SentenceTransformer = None,
    mini_check_model=None, 
    **kwargs,
) -> Dict[str, float]:
    """Compute measures of diversity between original and paraphrase."""
    res = {}

    words_orig = word_tokenize(original)
    common_words = Counter(words_orig) & Counter(word_tokenize(paraphrase))
    res["diversity/num_common_words"] = sum(common_words.values()) / len(words_orig)
    res["diversity/levenshtein_similarity"] = (
        distance.DamerauLevenshtein.normalized_similarity(
            original, paraphrase, processor=utils.default_process
        )
    )
    # ^note: Calculates the minimum number of insertions, deletions, substitutions
    # of characters required to change one sequence into the other
    if emb_model and isinstance(emb_model, SentenceTransformer):
        orig_emb = emb_model.encode(sent_tokenize(original))
        para_emb = emb_model.encode(sent_tokenize(paraphrase))

        similarity = orig_emb @ para_emb.T
        res["diversity/semantic_similarity_mean"] = float(similarity.mean())
        res["diversity/semantic_similarity_max"] = float(similarity.max())
        res["diversity/semantic_similarity_min"] = float(similarity.min())
        
    if mini_check_model and isinstance(mini_check_model, MiniCheck):
        original_sentences = [s.strip() for s in nltk.sent_tokenize(original) if s.strip()]
        paraphrase_sentences = [s.strip() for s in nltk.sent_tokenize(paraphrase) if s.strip()]
        if len(original_sentences) > 1:
            import pdb; pdb.set_trace()
            raise NotImplementedError("Not implemented yet")
        
        orig_sent = original_sentences[0]
        para_sent = paraphrase_sentences[0]
        pred_label, raw_prob, _, _ = scorer.score(docs=[orig_sent, para_sent], claims=[para_sent, orig_sent])
        res["diversity/entailment__minicheck"] = 1 if sum(pred_label) > 1 else 0
        res["diversity/entailment__minicheck_scores"] = raw_prob
    return res


@cache
def score_uncertainty(
    original: str, paraphrase: str, unc_model: CertaintyEstimator, prefix="uncertainty/", **kwargs
) -> Dict[str, float]:
    def aggr_prob_across_sentences(text: str):
        res = 0
        res_greedy = 0
        sentences = sent_tokenize(text.strip())

        for sentence in sentences:
            preds = unc_model.predict(sentence, get_processed_output=False)[0][0]
            # ^Note: Array shape num_unc_classes x categories for each class
            greedy_mask = np.zeros_like(preds, dtype=np.int8)
            greedy_mask[np.arange(preds.shape[0]), np.argmax(preds, axis=1)] = 1
            res_greedy += greedy_mask

            preds = stable_softmax(preds)
            res += preds

        return res / len(sentences), res_greedy / len(sentences)

    res = {}
    if isinstance(unc_model, CertaintyEstimator):
        norm_orig, norm_orig_greedy = aggr_prob_across_sentences(original)
        norm_para, norm_para_greedy = aggr_prob_across_sentences(paraphrase)
        # Uncertainty measures: absolute error and kl-divergence
        abs_error = norm_para - norm_orig
        kl_div = entropy(pk=norm_orig, qk=norm_para, axis=1, keepdims=True)

        for i, category in enumerate(CertaintyEstimator.UNCERTAINTY_CATEGORIES):
            res[f"{prefix}AE__{category.lower()}"] = float(np.abs(abs_error[i]).sum())
            res[f"{prefix}MAE__{category.lower()}"] = float(np.abs(abs_error[i]).sum() / len(CertaintyEstimator.UNCERTAINTY_ASPECT_MAPPING))
            res[f"{prefix}KL_div__{category.lower()}"] = float(kl_div[i, 0])

            for (j,category_pred) in CertaintyEstimator.UNCERTAINTY_ASPECT_MAPPING.items():
                res[f"{prefix}ER__{category.lower()}__{category_pred}"] = float(norm_para[i, j] - norm_orig[i, j])

        # Compute greedy counts (original metric)
        for i, category in enumerate(CertaintyEstimator.UNCERTAINTY_CATEGORIES):
            for (j,category_pred,) in CertaintyEstimator.UNCERTAINTY_ASPECT_MAPPING.items():
                res[f"{prefix}norm_counts/{category.lower()}__{category_pred}"] = (
                    norm_para_greedy[i, j] - norm_orig_greedy[i, j]
                )
    else:
        raise NotImplementedError("This is not supported yet.")
    return res


# -----------------------------
# Metric aggregation
# -----------------------------
def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if k in d and not isinstance(d[k], str)]
        out[k] = float(sum(vals) / max(1, len(vals))) if len(vals) else float("nan")
    return out


def score_run_against_original(fn, run: RunResult) -> List[Dict[str, float]]:
    """
    Returns one metrics dict per paraphrase turn, each compared against run.input_text.
    """
    original = run.input_text
    per_turn = [fn(original, original)]

    for o in run.outputs:
        per_turn.append(fn(original, o))
    return per_turn


class Evaluator:
    def __init__(self, semantic_similarity, uncertainty_predictor, length: bool = True):
        print("Setting up semantic similarity metrics")
        self.setup_semantic_similarity_metric(**semantic_similarity)
        print("Setting up uncertainty metrics")
        self.setup_uncertainty_metric(**uncertainty_predictor)
        self.length = length

    def setup_semantic_similarity_metric(self, model_name=None, device=None):
        if model_name:
            self.emb_model = SentenceTransformer(model_name)
            self.emb_model.eval()
            self.emb_model.to(device)
        else:
            self.emb_model = None

    def setup_uncertainty_metric(self, model_name=None, device: str = None):
        if model_name:
            self.unc_model = CertaintyEstimator(
                model_name, cuda=device.lower() == "cuda"
            )
        else:
            self.unc_model = None

    def compute_scores(self, original, paraphrase) -> Dict[str, float]:
        metrics = {}
        metrics_len = score_length(original=original, paraphrase=paraphrase)
        metrics.update(metrics_len)

        metrics_div = score_diversity(
            original=original, paraphrase=paraphrase, emb_model=self.emb_model
        )
        metrics.update(metrics_div)

        metrics_unc = score_uncertainty(
            original=original, paraphrase=paraphrase, unc_model=self.unc_model
        )
        metrics.update(metrics_unc)
        return metrics
