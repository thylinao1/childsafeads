# ChildSafeAds 2026: system and analysis code

Code for the system entered as `thylinao` in the [ChildSafeAds shared
task](https://www.codabench.org/competitions/17595/) (NLLP @ EMNLP 2026), and for the
measurements reported in the accompanying system description paper.

The task asks three questions about sponsored segments in YouTube videos likely to reach
children: what kind of offer is promoted (ST1), which product categories apply (ST2), and which
advertising-law concerns the segment raises (ST3). Systems are scored by the mean of three
macro-F1 columns.

## The data is not here

Competition Terms item 5 states: *"Do not redistribute the dataset outside your team. Direct
others to this competition instead."* This repository therefore contains **no dataset files, no
gold labels, no model probability files and no fold assignment**, and instance identifiers have
been removed from the documentation. Request access from the task organisers via the competition
page above, then place `train.jsonl`, `dev.jsonl` and `test.jsonl` under `data/`.

For the same reason, files that `METHOD.md` describes but that derive from the dataset
(`preds/`, `preds_test.jsonl`, `eval/folds.json`, `eval/majority_dev.jsonl`,
`decide/refit_eval.log`) are not included; the pipeline regenerates all of them once the data
is in place.

## What is here

| Path | Contents |
|---|---|
| `train/m0/` | TF-IDF baselines |
| `train/m1-encoder/` | DeBERTa-v3-large and ModernBERT-large fine-tuning |
| `train/m2q/` | Qwen3-14B LoRA sequence classifier |
| `train/m3q32/` | Qwen3-32B zero-shot lane, prompts and vLLM inference |
| `train/s3s/` | level-2 stackers |
| `decide/` | decision layer, fitted parameters, advertiser-precedent rule |
| `eval/` | scorer pinned to the official convention, channel-cluster bootstrap |
| `clinic/` | the analyses reported in the paper |
| `paper/latex/` | paper source and figures |

## Reproducing the paper's measurements

Each analysis is one command once the data and member predictions are in place.

```bash
python3 clinic/presence_artefact.py    # the divisor variance result
python3 clinic/level_ablation.py       # data access levels L1-L4
python3 clinic/member_ablation.py      # leave-one-out ensemble anatomy
python3 clinic/advertiser_overlap.py   # advertiser recurrence across splits
python3 clinic/testsize_noise.py       # standard error at evaluation size
python3 clinic/hand_label_audit.py     # recovers the hand-set labels from submitted archives
python3 clinic/per_class_report.py --oof-preds preds_oof.jsonl --dev-preds preds_dev.jsonl --latex
                                       # per-class P/R/F1 and the ST1 confusion matrix (paper Appendix C)
```

`eval/local_scorer.py` implements present-label-set macro-F1 and pins itself against the
organisers' published baseline row. `METHOD.md` describes the system and the metric analysis in
full; `RESULTS.md` records what each submission scored and what was measured and not kept.

## Submitted predictions

The entry submitted to the evaluation phase contained labels that were not produced by this
system, which conflicts with Competition Terms item 6. This was raised with the organisers after
the phase closed. `METHOD.md` section 7.5 gives the complete account, and
`clinic/hand_label_audit.py` reconstructs it from the submitted archives. The score reported as
the system's own is 0.6537, the last entry containing no hand-set label.

The edits were possible because the submitted file was assembled by hand from an emitted one.
`generate_submission.py --from-manifest` now verifies the pinned configuration
(`clinic/config_manifest.py verify`), re-emits the split from the pinned member probabilities
and `decide/params.json`, refuses to package a predictions file unless every row is identical
to that emission, and writes `<out>.provenance.json` with the archive's sha256 and the manifest
note, so that an uploaded file can be tied back to the configuration that produced it:

```bash
python3 decide/decision_layer.py emit --members m1d,m1m,m3q32A,m0s,s3s,m2qS2 --split test \
  --out preds_test_rebuilt.jsonl
python3 generate_submission.py --preds preds_test_rebuilt.jsonl --split test \
  --out submission_test_rebuilt.zip --from-manifest
```

## Licence

Code is released under the MIT Licence (`LICENSE`). It carries no dataset, and the dataset's own
CC BY-NC-SA 4.0 terms are unaffected by this repository.
