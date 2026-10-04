# Selective Cross-Model Verification for ERC

Code and artifacts for the ESWA submission *"Selective Cross-Model
Verification for Emotion Recognition in Conversations: When and Why
Heterogeneous Review Helps"*.

A fine-tuned primary LLM answers; a second, independently trained LLM from a
**different model family** is invoked only on the low-confidence tail, and a
disagreement is adopted only when the reviewer's label-sequence posterior
exceeds the primary's by a margin. On MELD (8 paired seeds) this improves
weighted-F1 by **+0.49 ± 0.46** points (paired *t*: p = 0.019) at **1.27×**
expected forward-pass cost; the frozen gate transfers zero-shot to IEMOCAP
(+1.21 ± 0.42, 3 seeds, p = 0.037).

## Repository layout

```
src/debate_erc/        library code (agents, deliberation, training, eval)
scripts/               training / evaluation / analysis entry points
configs/               experiment configs (YAML)
paper_eswa/            LaTeX source, figures, references
outputs/               per-utterance prediction records & aggregate results
```

## Reproduction map (paper table → artifact)

| Paper item | Artifact |
|---|---|
| Table 1 (8-seed main) | `outputs/selective_hetero/qwen7b*/test_records.jsonl`, `scripts/aggregate_8seeds_uniform.py` |
| Table 2 (positioning) | external reported numbers, see `paper_experiments_20261003.md` §26 |
| Fig. 1 (scenario) | `scripts/make_scenario_figure.py` |
| Fig. 2 (Pareto) | `scripts/make_paper_figures.py` |
| Fig. 3 (risk-coverage) | `scripts/make_paper_figures.py` |
| Fig. 4 (calibration) | `scripts/calibration_analysis.py` → `outputs/calibration/` |
| §5.2 controls (SC/debate/1.5B) | `outputs/self_consistency/`, `outputs/selective_hetero/` |
| §5.3 cross-family/-dataset | `outputs/selective_hetero/cross/`, IEMOCAP runs |
| §5.4 Pareto sweep | `outputs/selective_hetero/` τ grid |
| §5.5 abstention | `outputs/abstention/selective_prediction.json` |
| §5.6 case study | `outputs/selective_hetero/qwen7b/test_records.jsonl` |

## Environment

- Python 3.10, conda env `instructerc`
- torch ≥ 2.7 (cu128; required for sm_120 GPUs), transformers ≥ 4.45,
  peft 0.14, trl

## Quick start (inference-only audit of the main result)

```bash
python scripts/aggregate_8seeds_uniform.py   # reproduces Table 1 from records
python scripts/calibration_analysis.py       # reproduces Fig. 4 + §25 metrics
```

Full training requires the LLaMA2-7B base weights (Meta license) and MELD
(IEMOCAP requires its own license agreement). LoRA adapters are released on
Hugging Face Hub; see `paper_eswa/repo_release_checklist.txt` for the release
manifest.

## Citation

See `paper_eswa/` for the manuscript. License: code MIT (see LICENSE);
datasets and base-model weights remain under their upstream licenses.
