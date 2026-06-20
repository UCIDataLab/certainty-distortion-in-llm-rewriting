# From 'May' to 'Is': Certainty Distortion in Language Model Rewriting ([preprint](https://arxiv.org/pdf/2606.07951))

**Catarina G Belem, Shang Wu, Hongyu Yao, Mark Steyvers, Sameer Singh, Padhraic Smyth**

## Abstract

Humans increasingly turn to Language Models (LMs) in ways that shape beliefs and drive decisions, including discussing, rewriting, and summarizing information from scientific articles, news, and medical reports. However, in these domains, where how confidently a claim is expressed matters, little is known about whether LMs faithfully preserve it. In this work, we investigate *certainty distortion* in LMs, defined as meaningful changes in expressed certainty when semantic content is preserved. We propose an LM-based evaluation metric that is consistent with population-level judgments of certainty. Using this metric, we characterize certainty distortion across different sizes and families of models in the context of scientific and medical communication tasks. Our results show that certainty distortion affects up to 75% of LM outputs and is systematically asymmetric in rewriting tasks, with most LMs being 1.5–2× more likely to increase the expressed certainty than to decrease it. These effects can compound over repeated paraphrasing: in the medical domain, Haiku 4.5 increases certainty of 20% of examples after a single iteration, increasing to 40% after five iterations. Prompt-based interventions reduce overall certainty distortion but do not eliminate it. Together, these findings reveal a general bias toward inflating expressed certainty, with direct implications for users who rely on LMs in high-stakes domains.

---

## Key Findings

- **Distortion is pervasive.** Certainty distortion affects 30–75% of LM outputs across scientific and medical rewriting tasks, even in simple sentence-level paraphrasing.
- **LMs systematically inflate certainty.** Most models are 1.5–2× more likely to increase expressed certainty than to decrease it. Hedges like *may* and *suggests* are routinely dropped in favor of assertive language like *is* and *demonstrates*.
- **Task matters.** News rewriting amplifies distortion significantly more than paraphrasing. Models adopting journalism-style framing (e.g., "Researchers have discovered") strip hedges and present findings as established.
- **The broken telephone effect.** In the medical domain, certainty inflation compounds across iterations — doubling from ~20% to ~40% after five paraphrasing rounds. Scientific domain distortion mostly plateaus after the first rewrite.
- **Scale doesn't fix it.** Larger models distort less within a family, but no model achieves below 11% distortion on medical tasks or 39.5% on scientific news rewriting.
- **Prompting helps, but not enough.** Adding explicit *preserve-certainty* instructions reduces distortion and nearly neutralizes the directional bias, but does not eliminate it.

---

## Repository Contents

```
.
├── data/
│   ├── human_annotations/     # Prolific annotator responses (n=129) with pairwise certainty judgments
│   ├── spiced/                # Filtered SPICED sentence pairs used in experiments
│   └── mimic_cxr/
│       └── filter.py          # Script to reproduce our MIMIC-CXR sentence/document selection
│                              # ⚠️  Raw MIMIC-CXR data requires credentialed access via PhysioNet
│                              #     See: https://physionet.org/settings/credentialing/
├── generations/
│   ├── sentence_level/        # Model outputs for paraphrase and rewriting tasks (SPICED, MIMIC-CXR)
│   └── document_level/        # Model outputs for document-level tasks (Academic Papers, MIMIC-CXR)
├── configs/                   # Decoding configurations and model settings used across experiments
├── prompts/                   # All generation and evaluation prompts (see Appendix C & D of paper)
│   ├── generation/
│   └── evaluation/
├── scripts/                   # End-to-end commands to reproduce all experiments
└── src/                       # Source code for generation, evaluation, and analysis
```

### MIMIC-CXR Access

The MIMIC-CXR dataset is protected by a Data Use Agreement (DUA) and requires credentialed access through PhysioNet. To use our filtering script, you must first obtain access at [https://physionet.org/settings/credentialing/](https://physionet.org/settings/credentialing/).

---

## Evaluation Metric

We measure certainty distortion using a **pairwise LLM-as-a-judge** approach (`gpt-5.4-mini`, T=0). Each source–output pair is compared twice (with reversed input order) to control for positional bias. The judge selects from a 5-point scale (*Clearly Original* → *Clearly Modified*), and predictions are aggregated into a canonical label. This metric correlates with human population consensus more strongly than the average individual annotator (Kendall τ_B = 0.47 vs. 0.34) and outperforms fine-tuned BERT-based baselines.

---

## Citation

```bibtex
@misc{belem-2026-mayiscertaintydistortion,
      title={From `May' to `Is': Certainty Distortion in Language Model Rewriting}, 
      author={Catarina G Belem and Shang Wu and Hongyu Yao and Mark Steyvers and Sameer Singh and Padhraic Smyth},
      year={2026},
      eprint={2606.07951},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.07951}, 
}
```
