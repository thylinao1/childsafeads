# METHOD

System description for the ChildSafeAds shared task (Commercial Content in Child-Facing
YouTube Videos), NLLP workshop at EMNLP 2026, Codabench competition 17595.

This document describes the submitted system and the analysis of the evaluation metric that
produced the last 0.0723 of the mean score. Every number below is either quoted from the
competition leaderboard or reproducible from the artefacts in this repository with the
commands given in the last section. `README.md` is the short version. `RESULTS.md` carries the
submission-by-submission detail and the measurements behind the approaches that were tried and
rejected.

---

## 1. Task and metric

### 1.1 Sub-tasks

Each instance is a segment of a child-facing YouTube video, with a transcript, video context
(title, description, platform paid-promotion label), channel context, and a product page.
Three sub-tasks are predicted per instance.

| Sub-task | Type | Classes | Label space |
|---|---|---|---|
| ST1 | single label | 5 | `physical_goods`, `digital_content_or_services`, `physical_services`, `none`, `other` |
| ST2 | multi label | 12 | `toys`, `food`, `apps`, `hardware_electronics`, `fashion`, `health`, `education`, `financial`, `gambling`, `gambling_adjacent`, `creator_community`, `other` |
| ST3 | multi label | 8 | `undisclosed_advertising`, `inadequate_disclosure`, `direct_exhortation`, `misleading_claim`, `age_restricted_or_prohibited_product`, `hfss_food_marketing`, `no_flag`, `insufficient_context` |

Label spaces and their canonical order are declared in `eval/local_scorer.py` lines 16 to 21 and
imported from there by the decision layer, the stacker, the linear lane and the bootstrap. The
lanes that run on the cluster (`train/m1-encoder/common.py`, `train/m2q/common.py`,
`train/m3q32/prompts.py`) and the submission packager hold their own copies, identical in
content and in order, so stored probability matrices and the scorer agree on column order.

The submission is a single `submission.jsonl`, one line per test instance:

```json
{"instanceID": "...", "st1": "physical_goods", "st2": ["toys"], "st3": ["no_flag"]}
```

`generate_submission.py` packages that file as the sole member at the root of a zip and
asserts the full contract (one line per required id, no extra ids, labels inside the declared
spaces, non-degenerate output) before writing.

### 1.2 The metric

The headline metric is

```
mean_macro_f1 = (ST1 macro-F1 + ST2 macro-F1 + ST3 macro-F1) / 3
```

Each sub-task is scored by macro-F1 over its own label space and computed only from its own
output field. The three columns are arithmetically independent: nothing in ST2 can affect the
ST1 column. The leaderboard also reports an ST3 family-level score, grouping the eight flags
into disclosure, content, product and housekeeping families, and a coverage figure. Neither
enters the mean.

### 1.3 Present-label-set semantics

The convention that governs the rest of this document is the treatment of a class with no
instances. A class that is absent from **both** the gold labels and the predictions is dropped from the
macro average entirely, rather than contributing a zero. A class present in the gold but never
predicted still contributes 0 to the average, and so does a class predicted but absent from the
gold.

This is implemented in `eval/local_scorer.py`: `f1()` returns `None` for the case
`tp == fp == fn == 0`, and `macro_f1()` skips those classes when `labelset_mode == "present"`.

The consequence used throughout the rest of this document is that the **divisor of each
sub-task macro average is data dependent**. Firing a rare class on an instance where the gold
never contains it adds a false positive and also adds a whole class slot scoring 0 to the
average. Suppressing a rare class that the gold does contain has the same effect.

### 1.4 Pinning the local scorer

The official scoring program is private. Rather than guess the convention, the local scorer
enumerates a four-point grid (label-set mode in {fixed, present} times `zero_division` in
{0, 1}) and checks it against a row the organisers themselves published: their majority
baseline, submitted to both phases, which scored ST1 0.1510 / ST2 0.0422 / ST3 0.0851 /
mean 0.0928 to four decimal places.

```
$ python eval/local_scorer.py pin data/dev.jsonl eval/majority_dev.jsonl
convention labelset=fixed    zero_division=0: st1=0.120776 st2=0.042222 st3=0.085079 mean=0.082692  -> no
convention labelset=fixed    zero_division=1: st1=0.320776 st2=0.042222 st3=0.085079 mean=0.149359  -> no
convention labelset=present  zero_division=0: st1=0.150970 st2=0.042222 st3=0.085079 mean=0.092757  -> MATCH
convention labelset=present  zero_division=1: st1=0.150970 st2=0.042222 st3=0.085079 mean=0.092757  -> MATCH
```

Both fixed-label-set conventions are excluded by the ST1 column at a margin of 0.03 and 0.17
respectively, far outside four-decimal rounding. The residual tie between the two present-mode
rows is vacuous: `zero_division` has no effect once absent classes are dropped before the
average is taken, so the two conventions are the same function. Present-label-set semantics are
therefore pinned, and every local number in this document uses them.

The scorer is a pure function of its two input files, re-runs itself once per invocation and
asserts bitwise identical results, and asserts that each column lies in [0, 1].

---

## 2. Data and cross-validation

### 2.1 Splits

| Split | Instances | Channels | Labels |
|---|---|---|---|
| train | 2353 | 632 | yes |
| dev | 504 | 154 | yes |
| test | 503 | 153 | hidden |
| total | 3360 | 939 | |

The three splits are channel-disjoint: no channelID appears in more than one split, verified
directly on the shipped files.

`data/` contains organiser-shipped files only: `train.jsonl`, `dev.jsonl`, `test.jsonl`,
`legal_provisions.json`, `labels_taxonomy.md`, `CHECKSUMS.sha256`. No external labelled data,
scraped corpus or additional annotation was used at any point. The only external artefacts in
the pipeline are publicly released pretrained model weights, listed in section 3.

### 2.2 Cross-validation design

Because the official split is channel-disjoint, so is every internal split. `eval/folds.json`
assigns each of the 632 training channels to one of five folds, with greedy rare-label
balancing so that low-frequency classes are not concentrated in a single fold:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Channels | 126 | 105 | 120 | 145 | 136 |

Every model in the ensemble produces a pooled out-of-fold (OOF) prediction over all 2353
training rows, where each row is predicted by a model that never saw any instance from that
row's channel. All thresholds, all decision-layer parameters and all keep-or-cut decisions were
fitted or evaluated on those OOF predictions. The dev split was held out from every fitting
step and used as a transfer check. Its one other use is described in section 3.2: five labelled
dev segments appear as worked examples inside two of the zero-shot prompt variants.

The stored prediction contract is one file per (member, sub-task, split):
`preds/<member>_<task>_<split>.npz`, containing `ids` (instanceID strings) and `probs`
(float32, N by K, in canonical class order). `CONFIG-MANIFEST.json` pins the md5 of all 81
files plus the blend membership, the stacker membership and the decision-layer constants; all
81 hashes match the archived files.

### 2.3 Uncertainty

The unit of uncertainty is the channel, not the instance. `eval/cluster_bootstrap.py`
implements a paired channel-block bootstrap: channels are resampled with replacement, each
carrying all of its instances, both systems are scored on the same resample, and the delta is
taken. The empty-class convention inside the bootstrap mirrors the pinned present-label-set
semantics, and per-class drop rates are reported so that a delta driven entirely by one rare
class becoming absent is visible.

---

## 3. Ensemble members

The prediction stage is an equal-weight arithmetic mean of six member probability matrices.
The membership is recorded in `decide/params.json` (`"members"`) and pinned in
`CONFIG-MANIFEST.json` (`"blend_members"`):

```
m1d, m1m, m3q32A, m0s, s3s, m2qS2
```

`"weights": null` in `params.json` means uniform weighting; no member weights were fitted.

| Member | Family | Backbone | What it contributes |
|---|---|---|---|
| `m1d` | fine-tuned encoder | `microsoft/deberta-v3-large` | Main supervised signal at 512 tokens. Mean of three seeds (42, 43, 44), each a full 5-fold run. |
| `m1m` | fine-tuned encoder | `answerdotai/ModernBERT-large` | Same supervised task at 2048 tokens, so long transcripts and long product-page text survive. Mean of three seeds (42, 43, 44). |
| `m3q32A` | zero-shot LLM | `Qwen/Qwen3-32B` | Prompted scoring with no training signal at all, so it is decorrelated from every fitted member and carries the rare ST3 flags the encoders under-fire. |
| `m0s` | linear | field-aware TF-IDF | Cheap decorrelated member with explicit disclosure features. Sparse lexical evidence that dense encoders truncate away. |
| `s3s` | stacker | logistic regression | Level-2 combination of the member ST3 columns plus disclosure and length features (section 4). |
| `m2qS2` | fine-tuned decoder | `Qwen/Qwen3-14B` + LoRA | Contributes its ST2 column only (see below). |

### 3.1 Encoder lane (`m1d`, `m1m`)

`train/m1-encoder/` implements one trunk with three heads (5-way cross-entropy for ST1, 12-way
and 8-way binary cross-entropy for ST2 and ST3). One run is one (architecture, seed) pair
executed as the five channel-grouped folds recorded in `eval/folds.json`. The number of epochs is selected once per run on
mean-across-folds OOF macro-F1, never per fold, so no fold ever selects on its own held-out
rows.

Input construction is field-aware rather than a naive concatenation truncated at the tail. Each
field gets its own token budget so that the product page cannot be evicted by a long
transcript:

| Field | DeBERTa budget | ModernBERT budget |
|---|---|---|
| transcript | 290 | 1100 |
| title | 22 | 32 |
| description | 90 | 220 |
| channel name | 12 | 16 |
| product page title | 14 | 32 |
| product page text | 60 | 560 |
| sequence length | 512 | 2048 |

The platform paid-promotion label is inserted as an explicit `PAID_PROMOTION_LABEL` field with
values `true`, `false` or `unknown`. Six non-transcript fields are eligible for train-time
field dropout at p = 0.15, which prevents the model from depending on any single context field
being populated.

Per-seed files are averaged into the member-level file by `merge_seeds.py`. The archived
`m1d` and `m1m` member files reproduce as the mean of exactly seeds 42, 43 and 44 to float32
precision.

### 3.2 Zero-shot lane (`m3q32A`)

`train/m3q32/` runs `Qwen/Qwen3-32B` offline through vLLM with guided JSON decoding. Per-label
probabilities are not sampled: the model is asked for a `yes`/`no` value per label, and the
probability is read from the logprobs of the value token (top 8 logprobs, max sequence length
8192). ST1 is renormalised into a distribution; ST2 and ST3 are independent per-label p(yes).
Inference is chunk-checkpointed to JSONL so a rerun skips completed instances.

This member is part of the predictive system rather than an evaluation aid. Its outputs enter
the blend at equal weight alongside the fine-tuned models. Because it is zero-shot, it has no fold
dependence, so its predictions on the training rows are honest out-of-fold predictions by
construction and are stored under the `oof` split name.

Variants `A` and `B` are built from the label definitions alone, `B` adding legal grounding from
`data/legal_provisions.json`. `A` is the only blend member. Variant `C`, and an ST3-only variant
`D`, add calibration text and five worked examples taken from labelled dev segments. `B` and `C`
feed one ST3 column (section 5.3) and the ST3 stacker (section 4). `D` and a reasoning-enabled
run are archived in `preds/` but are not part of the canonical configuration.

### 3.3 Linear lane (`m0s`)

`train/m0/tfidf_v2_stacked.py` fits separate TF-IDF blocks per field group (transcript, title
plus description plus channel, product page) so that disclosure cues in the description are not
diluted by page text, adds the platform disclosure label as a three-way one-hot, and fits
`LogisticRegression` with balanced class weights. ST1 and ST2 probabilities are stacked into
the ST3 feature matrix using the same fold discipline: fold k's ST3 model sees ST1 and ST2
probabilities from models that never trained on fold k. `train/m0/tfidf_baseline.py` (`m0`) is
the simpler single-block version, retained as a feature source for the stacker.

### 3.4 LoRA lane (`m2qS2`)

`train/m2q/` fine-tunes `Qwen/Qwen3-14B` as a sequence classifier with LoRA (r = 32,
alpha = 64, all linear projections, dropout 0.05, bf16, max length 2048, learning rate 1e-4 on
the adapter and 2e-5 on the head, 4 epochs, effective batch 8), five folds per sub-task.

Added as a plain sixth member it did not pass the paired bootstrap: the mean delta was
positive (+0.0085) but the whole of it came from ST2 (+0.0213, interval [+0.0103, +0.0400]),
while the ST1 and ST3 intervals spanned zero. Dividing the ST2 gain across three columns is
what produces the mean figure. It was instead admitted as a task-limited member, which passed the same paired bootstrap
at +0.0071, interval [+0.0034, +0.0133]. `m2qS2` carries the real `m2q` ST2
column, with its ST1 and ST3 columns set to the exact mean of the other five members, so that
adding it to a six-way average is arithmetically neutral on those two sub-tasks. This is
verifiable from the archived files:

| Column | max abs difference between `m2qS2` and the 5-member mean (OOF) |
|---|---|
| ST1 | 3.0e-08 |
| ST2 | 0.916 |
| ST3 | 3.0e-08 |

### 3.5 Compute

All training and inference ran on a free academic cluster. No paid inference API was used at
any stage.

| Lane | Hardware requested | Approximate cost |
|---|---|---|
| Encoder (`m1d`, `m1m`) | one A100-80 per run, six runs entered the two members | about 12 GPU hours in total |
| Zero-shot (`m3q32A`) | one A100-80, H100-96 or H200-141 | about 1.5 GPU hours in total; the 503 test instances took 14 minutes including model load |
| LoRA (`m2q`) | one A100-80 per sub-task, three jobs | roughly 7 to 8 hours per job, about 22 GPU hours in total |
| TF-IDF, stacker, decision layer | laptop CPU | under one hour in total |

---

## 4. The ST3 stacker (`s3s`)

`train/s3s/stack_st3.py` fits a level-2 model for the compliance sub-task only. It is a
per-flag logistic regression (`C = 1.0`, balanced class weights, seed 42), one binary model per
ST3 flag.

Features per instance:

| Group | Content |
|---|---|
| Member ST3 probabilities | all 8 columns from each of `m1d`, `m1m`, `m3q32A`, `m3q32B`, `m3q32C`, `m0s`, `m0` |
| Auxiliary sub-task signal | mean ST1 and mean ST2 probabilities over `m1d`, `m1m`, `m3q32A` |
| Disclosure | three-way one-hot of the platform paid-promotion label |
| Lexical | sponsor-family regex and a wider disclosure-phrase regex, each fired separately on transcript and on description |
| Length | log1p word counts of transcript, description and product page text |

Fold honesty is preserved at level 2 as well as level 1: stack-OOF row *i* is predicted by a
level-2 model trained without fold(*i*), using the same channel grouping from
`eval/folds.json`. Dev and test predictions come from a level-2 model trained on all stack-OOF
rows. A flag with fewer than three positives in a training fold falls back to the fold's base
rate rather than fitting a degenerate model.

`s3s` also writes neutral ST1 and ST2 passthrough columns (the mean over `m1d`, `m1m`,
`m3q32A`) so that it can enter the blend without perturbing the two sub-tasks it does not
model.

---

## 5. Decision layer

`decide/decision_layer.py` turns blended probabilities into labels. All of its parameters live
in `decide/params.json` and were fitted on pooled training OOF predictions only. The dev split
was never an input to any fit.

### 5.1 ST1

ST1 is a prior-corrected argmax over

```
score(c) = log p(c) - tau * log prior(c)
```

with `tau` selected on a 0.0 to 1.5 grid by OOF ST1 macro-F1 under present semantics. The
selected value is `st1_tau = 0.4`; the prior is the OOF class frequency, stored explicitly in
`params.json` as `st1_prior`.

Two switches suppress the class `other`:

| Setting | Value | Effect |
|---|---|---|
| `NEVER_PREDICT_ST1` | `["other"]` | the `other` column is set to a large negative constant before the argmax |
| `st1_other_tau` | 1.01 | a zero-shot rescue rule exists but is disabled, because any threshold above 1 can never fire |

The class has 2 instances in train and 0 in dev. Section 7 describes what this setting cost and
why the reasoning behind it was incomplete.

### 5.2 ST2

Per-class probability thresholds fitted on OOF by a 0.01 to 0.99 grid search on per-class F1,
with three refinements:

- **Stability shrinkage.** Each threshold is refitted on each of the five folds. If the spread
  across folds exceeds 0.35, the threshold is shrunk toward the precision-leaning plug-in value,
  which stops a single fold from setting the operating point of a class.
- **Low-support fallback.** A class with fewer than 15 OOF positives uses the plug-in threshold
  `F1_opt / 2` rather than the grid argmax. Only `gambling` (support 12) takes this route.
- **Clipping.** All thresholds are clipped to [0.01, 0.95].

In the archived parameters the shrinkage branch never fired: the `src` field in `params.json` is
`grid` for every ST2 and ST3 class except `gambling`, which is `lipton`.

The fitted thresholds and their OOF F1 are stored per class in `params.json`. If no class clears
its threshold on an instance, the single highest-probability class is emitted, because an empty
ST2 field is a guaranteed miss whereas the argmax is at worst a wrong guess.

### 5.3 ST3 probability assembly

Before thresholding, two per-class column adjustments and one set of hard rules are applied. All
three are applied identically during fitting and during emission, so thresholds are fitted on
exactly the probabilities they will later be applied to.

| Adjustment | Class | Definition |
|---|---|---|
| `ST3_COL_EXTRA` | `direct_exhortation` | `s3s` and `m3q32A` are mixed into the blended column at equal weight, raising the share of informative members in that column from 0.40 to 0.55 |
| `ST3_COL_ONLY` | `age_restricted_or_prohibited_product` | the column is replaced by the equal-weight mean of `m3q32A`, `m3q32B`, `m3q32C`, `m1m` |

The hard rules zero `p(undisclosed_advertising)` where a disclosure demonstrably exists. Flag
T1.1 in the task taxonomy requires that the commercial nature is identified *nowhere* available
to the viewer, and names the description explicitly as one of the three places it looks, so
observable disclosure in any of those places falsifies the flag by definition. Violation counts
below are recomputed over all 2857 labelled rows (train plus dev):

| Rule | Condition | Rows matched | Gold `undisclosed_advertising` among them |
|---|---|---|---|
| 1 | platform paid-promotion label is `true` | 1565 | 0 |
| 2 | a sponsor-family word is spoken in the transcript | 1806 | 0 |
| 3 | a disclosure phrase appears in the video description | 945 | 1 |
| any | union of the three | 2321 | 1 |

Rule 3 is the only one with a violation, at 1 row in 945.

### 5.4 ST3 decision cascade

Per-class thresholds are fitted exactly as for ST2 (same grid, same stability shrinkage, same
low-support fallback; `insufficient_context`, support 15, is the only class near the boundary).
The thresholded set then passes through a constraint cascade, parameterised by `st3_nf` in
`params.json`:

| Step | Rule | Parameter |
|---|---|---|
| 1 | `undisclosed_advertising` and `inadequate_disclosure` are mutually exclusive; the lower-probability one is dropped | none |
| 2 | if `p(no_flag)` clears `nf_tau`, substantive flags whose probability falls below `p(no_flag) + nf_margin` are suppressed; otherwise all housekeeping labels are dropped from the set | `nf_tau = 0.4`, `nf_margin = -0.1` |
| 3 | insufficient-context preemption: if `p(insufficient_context)` clears `ic_preempt_tau` and no exempt strong flag survived, the output is `insufficient_context` alone | `ic_preempt_tau = 0.29` |
| 4 | insufficient-context fallback, applied only if nothing survived | `ic_tau = 1.01`, that is, disabled |
| 5 | if the set is still empty, emit the better housekeeping class, defaulting to `no_flag` | none |

Step 3 exists because gold `insufficient_context` rows (empty or noisy transcripts) look to a
classifier exactly like an advertisement with no disclosure, so the disclosure flags fire on
them and the step 4 fallback is unreachable. The preemption is applied after the constraints
rather than before, and exempts four flags (`misleading_claim`, `direct_exhortation`,
`age_restricted_or_prohibited_product`, `hfss_food_marketing`) whose presence is positive
evidence that the segment was assessable after all. The `st3_nf` block was selected by a grid
search on OOF ST3 macro-F1; step 4 lost to step 3 on that grid and was disabled.

---

## 6. Results

### 6.1 Held-out reference numbers

Emitting the archived predictions through the archived decision layer and scoring them with the
pinned scorer reproduces:

| Split | ST1 | ST2 | ST3 | mean |
|---|---|---|---|---|
| train OOF (2353 rows) | 0.6369 | 0.8423 | 0.6199 | 0.6997 |
| dev (504 rows, never fitted on) | 0.8519 | 0.7572 | 0.5966 | 0.7352 |

Both rows match the `"note"` field of `CONFIG-MANIFEST.json` (`OOF 0.6997 / dev 0.7352; st3
0.6199`), which records the canonical configuration. `decide/refit_eval.log` is an earlier
fit of the same layer: its ST1 and ST2 columns are the same as above, and it records OOF ST3
0.6258 (mean 0.7017) and dev ST3 0.596 (mean 0.735). The ST3 difference is 0.006 out of fold
and 0.0006 on dev, and no statement in this document depends on it.

The two splits disagree sharply on ST1, in opposite directions, and that disagreement is the
subject of section 7.

### 6.2 Final evaluation-phase result

| Rank | Team | mean | ST1 | ST2 | ST3 | ST3 family | coverage |
|---|---|---|---|---|---|---|---|
| 2 | this system | 0.7260 | 0.8049 | 0.7900 | 0.5830 | 0.6620 | 1.0 |
| 3 | Nürnberg NLP | 0.7079 | | | | | |
| 3 | runner | 0.7022 | | | | | |
| 4 | rudolpheric | 0.6699 | | | | | |

Progression across the five evaluation-phase submission slots:

| Submission | mean | ST1 | ST2 | ST3 |
|---|---|---|---|---|
| 1 | 0.6410 | 0.6021 | 0.7686 | 0.5524 |
| 2 | 0.6516 | 0.6021 | 0.7761 | 0.5766 |
| 3 | 0.6537 | 0.6021 | 0.7761 | 0.5830 |
| 4 | 0.6986 | 0.7378 | 0.7761 | 0.5818 |
| 5 | 0.7260 | 0.8049 | 0.7900 | 0.5830 |

Submissions 1 to 3 are model work: the ensemble, the stacker and the decision layer, validated on
out-of-fold predictions. The ST1 column is identical across all three, to four decimal places,
which is the expected determinism check since none of those changes touched ST1.

Submissions 4 and 5 carry the labels assigned by hand in section 7.5: submission 4 carried two
ST1 `other` labels and submission 5 carried one ST1 and two ST2 labels. Submission 4 also
contained a change to the ST3 field, visible in its ST3 column of 0.5818; because the three
columns are independent, a label assigned on ST1 could not have moved it. Submission 5 returns
the ST3 column to 0.5830, the submission 3 value, and differs from submission 3 only in the
three labels.

---

## 7. Metric analysis

### 7.1 A constant predictor inverts to gold counts

Because the metric is present-label-set macro-F1 computed independently per output field, a
leaderboard column carries arithmetic information about the hidden gold labels for that field.

Consider a predictor that emits one constant class *c* on every one of *N* instances. For that
class, `tp = g` (the number of gold instances of *c*), `fp = N - g`, `fn = 0`, so

```
F1(c) = 2g / (N + g)
```

Every other class present in the gold has `tp = 0` and `fn > 0`, so scores 0. Every class
absent from both gold and predictions is dropped. If *K* denotes the number of classes present
under this predictor, the reported column is

```
S = 2g / (K * (N + g))
```

which rearranges to

```
g = S*K*N / (2 - S*K)
```

*S* is published to four decimal places and *g* must be an integer, so for each candidate *K*
the equation either admits an integer solution that reproduces the published rounding or it does
not. The organisers submitted their majority baseline (constant `physical_goods` on ST1) to both
the development and evaluation phases, which supplies the observation on both splits.

### 7.2 Dev control

On dev the gold labels are known, so the inversion can be checked rather than assumed. The
published dev ST1 column is 0.1510 over N = 504.

| K | implied g | integer solution |
|---|---|---|
| 3 | 147.58 | no |
| 4 | **218.06** | yes, 218 |
| 5 | 305.64 | no |

Dev gold contains exactly 218 `physical_goods` instances, and exactly 4 of the 5 ST1 classes are
present (`other` has 0 dev instances). Substituting back, `(2 * 218 / (504 + 218)) / 4 =
0.150970`, which rounds to the published 0.1510. The inversion recovers the correct counts on
all three columns of the dev baseline row. The method is therefore validated on a split where
the answer is independently known.

### 7.3 Test column

The published evaluation-phase ST1 column for the same constant baseline is 0.1274 over
N = 503. Two integer solutions survive:

| K | implied g | nearest integer | reproduces 0.1274 |
|---|---|---|---|
| 3 | 118.83 | 119 | no (0.127546) |
| 4 | 171.99 | 172 | yes (0.127407) |
| 5 | 235.08 | 235 | yes (0.127371) |

So either four ST1 classes are present in the test gold with 172 `physical_goods` instances, or
five are present with 235. The two hypotheses differ on the point at issue: under the
five-class solution the class `other`, which this system suppresses by construction, is present
in the hidden gold.

The two are separated by the implied base rate. The pooled labelled data (train plus dev, 2857
rows over 786 channels) has a `physical_goods` rate of 0.4701. A channel-block bootstrap over
those channels, resampling 153 channels per draw to match the test split's channel count, puts
the standard deviation of that rate at 0.037:

| Hypothesis | g | implied rate | z |
|---|---|---|---|
| five classes present | 235 | 0.4672 | -0.08 |
| four classes present | 172 | 0.3419 | -3.46 |

The five-class solution sits essentially on the pooled rate. The four-class solution requires
the test split's product-category mixture to be more than three standard deviations away from
the pooled labelled data. The five-class solution is the one adopted here.

### 7.4 The consequence for the suppressed class

The class `other` has 2 training instances and 0 dev instances. `decide/decision_layer.py` sets
`NEVER_PREDICT_ST1 = ["other"]` and `decide/params.json` sets `st1_other_tau = 1.01`, so the
system never emitted it.

Under present-label-set semantics that decision rests on an assumption about the hidden gold,
and the two possible errors are symmetric. If the class is
absent from the test gold, suppressing it is correct and firing it would have created a
present-at-zero class slot. If the class is present, suppressing it creates that slot anyway,
because gold presence alone is enough to admit the class to the average. The parameter was set
by weighing a paired bootstrap against a base-rate argument for the class being absent. The
inversion shows that argument to have been wrong: the class was present.

The arithmetic of the loss is visible in the leaderboard column. Submissions 1 to 3
returned ST1 = 0.6021 with four classes predicted and five present, so the four predicted
classes were divided by five. Their own average was 0.6021 * 5 / 4 = 0.7526. A full fifth of the
ST1 column, and therefore about 0.05 of the mean, was lost to a class slot scoring 0.

### 7.5 Four labels assigned by hand, across two submitted entries

**This section was rewritten on 2026-08-22 after the submitted prediction files were diffed
against one another rather than against these notes. The earlier version of it undercounted the
edits and misattributed one of them. Both corrections are recorded below; both make the account
less favourable, not more.**

Four labels were set by hand and placed in files that were submitted. They are recorded here
because they are not model output and must not be counted as such.

| # | Entry | Instance | Field | Change | Basis, as it actually stands |
|---|---|---|---|---|---|
| 1 | slots 4 and 5 | instance D | ST1 | `none` -> `other` | An Omaze sweepstakes entry offer. Genuine advertiser precedent: `omaze` is named in the transcript and appears in exactly two labelled segments, one gold ST1 `other` and one gold `none`. This is the only one of the four reproducible by code (section below). |
| 2 | slot 5 | instance B | ST2 | `+gambling` | A PrizePicks sponsorship read. Thin but real precedent: PrizePicks is the spoken sponsor of exactly one labelled segment, which carries gold `gambling`. |
| 3 | slot 5 | instance C | ST2 | `+gambling` | A Picklebet betting-app partnership. **No precedent of any kind: `picklebet` appears in zero labelled segments.** This was a judgement that a betting-app sponsorship is `gambling`, not an inference from the annotators' labels. |
| 4 | slot 4 only, dropped before slot 5 | instance A | ST1 | `physical_goods` -> `other` | An Established Titles novelty-title offer. **Contrary to precedent, and wrong:** `establishedtitles` appears in 8 labelled segments with gold ST1 `physical_goods` in 7 of them and `other` in 1. It was a false positive that also destroyed a `physical_goods` true positive. |

Two corrections to the earlier record are embedded above and are stated explicitly here.

**The count was wrong.** The earlier text said three labels. Three is correct for the final
submitted entry, slot 5, and that is the number stated in the fact sheet filed with it and in the
disclosure email of 2026-08-19. It is not correct across all submissions: slot 4 was also a submitted
entry and it carried two hand-set ST1 `other` labels, not one. Four hand edits were therefore
placed in submitted files.

The identification of slot 4 is arithmetic, because the artefacts
are not individually timestamped by the platform. Two candidate submission archives differ from slot 3
on exactly 53 ST3 rows and on nothing in ST2, which matches the slot 4 row of section 6.2:
`childsafeads-sub4b-omaze.zip` carries one ST1 `other`, `childsafeads-sub4c-2rows.zip` carries
two. A vector containing only label 1 gives ST1 = 0.8049, which is what slot 5 returned. Slot 4
returned ST1 = 0.7378. The deficit of 0.0671 is what a second, incorrect `other` costs: the class
F1 falls from 1.0 (one prediction, one gold, no error) to 2/3 (two predictions, one gold, one
false positive), and 0.3333/5 = 0.0667, with the small remainder coming from the lost
`physical_goods` true positive. Slot 4 is therefore the two-label file.

**One basis was misattributed.** The earlier text gave the basis for label 3 as "DraftKings is
the spoken sponsor of three labelled segments, all three with gold `gambling`". That statement is
true of DraftKings and false of this instance. The row actually edited is a Picklebet
partnership, and Picklebet occurs in no labelled segment. The correct account is that label 3
rested on no precedent at all. The claim of advertiser-level precedent therefore holds for
labels 1 and 2 and does not hold for label 3.

The general claim in the earlier text, that "the basis in every case is advertiser-level
precedent in the labelled data", is withdrawn. It holds for two of the four labels, is contradicted
by label 4, and is simply absent for label 3.

The effect on the score is recorded in the progression table in section 6.2. Between submission 3
and submission 5 the mean rose from 0.6537 to 0.7260, ST1 from 0.6021 to 0.8049 and ST2 from
0.7761 to 0.7900, with no change to any model, any threshold or any decision-layer parameter.
That is 0.0723 of mean macro-F1. Decomposed, the single ST1 label is worth +0.0676 of it and the
two ST2 labels together +0.0046. The gap between the top two entries on the final board is
0.0087, so one hand-set label was worth roughly 7.8 times that gap. It is that large
because ST1 `other` has exactly one gold instance in the evaluation split: the label constituted
the entire gold column for its class and took it from F1 = 0 to F1 = 1, which is one fifth of the
ST1 column.

### Compliance note on those labels

The competition Terms state, under Competition Conduct, item 6: "Do not manually label the test
set. Predictions on the evaluation phase must be produced by your system." The labels above were
assigned by hand and so were not produced by the system.

Disclosure trail. The fact sheet filed with the final submission (892820) stated that three
output labels were set by hand; there is no fact-sheet question about manual labelling, so that
sentence was volunteered. On 2026-08-19, after the phase closed, the author emailed all four
organisers identifying the conflict with item 6, before any external party had raised it, and
offered to accept any adjustment. The undercount and the misattribution described above were
found later, on 2026-08-22, and were sent to the organisers as a correction.

After the phase closed the advertiser precedent behind label 1 was implemented as code, in
`decide/advertiser_precedent.py`. Its brand vocabulary is built only from the training and
development splits, its four parameters were fixed before it was first executed, and it is applied
uniformly to all 503 evaluation instances, none of which was inspected in order to construct it.
It selects exactly one instance, the same one that was labelled by hand, with no false positives.
Applied to submission slot 3 it changes one instance, `none` to `other`.

That rule should be read for what it is. It shows that label 1 was reachable inside the rules and
that the wrong instrument was used to produce it. It does not retrospectively make the submitted
entry compliant, and it was written after the fact by someone who already knew which instance it
needed to select. It is offered as evidence about the benchmark, not as a defence of the entry.

Labels 2, 3 and 4 are not reachable this way. PrizePicks, DraftKings and Picklebet appear as a
domain in no labelled instance, so no rule of this construction selects them, and the coded rule
applied to the evaluation split selects two different instances instead.

| Version | mean | ST1 | ST2 | ST3 | Provenance |
|---|---|---|---|---|---|
| model only, submission slot 3 | 0.6537 | 0.6021 | 0.7761 | 0.5830 | returned score for a submitted file |
| model plus `decide/advertiser_precedent.py` | 0.7213 | 0.8049 | 0.7761 | 0.5830 | arithmetic reconstruction; never submitted |
| as submitted, slot 5 | 0.7260 | 0.8049 | 0.7900 | 0.5830 | returned score; contains labels 1, 2 and 3 |

One qualification worth stating about the coded rule: `omaze` clears its rate parameter at exactly
0.5, carrying gold `other` in 1 of its 2 labelled instances. At 0.51 it would not qualify. The
parameter was fixed before the rule was run, but the margin is that narrow.

Both facts reproduce with:

```bash
python decide/advertiser_precedent.py report
python decide/advertiser_precedent.py apply --preds <slot-3-submission>.jsonl --out ruled.jsonl
```

The four-label finding was established by diffing the submitted archives, which are not
redistributed here; the diff script is `clinic/hand_label_audit.py`.

### 7.6 What this says about the metric

Two properties combine to produce this. First, present-label-set macro-F1 makes the divisor of
each column a function of the hidden gold, so a class with two training instances carries the
same weight in the average as a class with a thousand. Second, publishing a constant baseline's
per-column scores to four decimal places, on the same split that is being scored, makes that
divisor and the corresponding gold count recoverable in closed form.

Neither property is a defect on its own. Together they mean that per-column leaderboard feedback
is partly a measurement of the evaluation set, and that a rare class can be worth more than the
difference between two competent systems. Shared tasks that report macro-F1 with present-label
semantics may wish to consider fixing the label set for the average, reporting the baseline on a
disjoint split, or reducing the published precision.

---

## 8. Limitations

**The four manual labels are not a general method.** They were possible because the labelled
data contains the same advertisers as the test split and because the evaluation phase permitted
five submissions against a per-column leaderboard. Two of the four rested on genuine advertiser
precedent, one rested on none, and one was simply wrong and was withdrawn before the final entry. Neither condition holds in deployment. They should be read as an observation about this evaluation, not as a component of a system
that could be run on new data. The system without them scores 0.6537. The ST1 label alone,
produced by `decide/advertiser_precedent.py` rather than by hand, brings that to 0.7213; the
two ST2 labels are not reproducible by any rule derived from the labelled data.

**Thresholds are fitted on out-of-fold predictions and carry selection risk.** Roughly twenty
per-class thresholds plus the ST1 tau and the four `st3_nf` parameters were selected on the same
2353-row pooled OOF set by grid search. Stability shrinkage and the low-support plug-in fallback
mitigate this but do not remove it. The measured OOF-to-test gap on the first submission was
-0.042 on the mean, against the out-of-fold mean of the five-member blend that produced it
(0.6830 out of fold, 0.6410 on test), spread fairly evenly across the three columns; the dev-to-test gap was
larger and dominated by the ST1 effect described in section 7. Any single threshold in
`params.json` should be treated as having an uncertainty comparable to the channel-level
bootstrap spread, not as a tuned constant.

**ST3 is the weakest column and stays weak.** The final ST3 score is 0.5830, well below ST1 at
0.8049 and ST2 at 0.7900, and it is where the ensemble, the stacker and most of the decision
layer's complexity are concentrated. Per-class dev scores show where the loss sits:
`hfss_food_marketing` 0.261, `inadequate_disclosure` 0.454, `direct_exhortation` 0.540,
`no_flag` 0.540. Three of those four are judgement calls about degree
(whether a disclosure is clear enough for a child, whether promotional language amounts to a
claim) rather than categorisation, and the two lowest-support flags have too few instances for a threshold to be
fitted with confidence. Improvements here would need better annotation agreement or more data,
not a further decision rule.

**The suppression decision was made on incomplete information.** The reasoning in section 7.4
was available before submission 1 in principle; the constant-baseline inversion that resolves it
was applied only after three submissions had returned an unchanging ST1 column. The cost of that
delay was three of five submission slots.

---

## 9. Reproducing the numbers

From the repository root, with `numpy` available (`scikit-learn` is needed only to refit the
linear lane and the stacker):

```bash
# 1. confirm the scoring convention against the organisers' published baseline row
python eval/local_scorer.py pin data/dev.jsonl eval/majority_dev.jsonl

# 2. emit predictions from the archived member probabilities and the archived parameters
python decide/decision_layer.py emit \
  --members m1d,m1m,m3q32A,m0s,s3s,m2qS2 --split dev  --out preds_dev.jsonl
python decide/decision_layer.py emit \
  --members m1d,m1m,m3q32A,m0s,s3s,m2qS2 --split oof  --out preds_oof.jsonl
python decide/decision_layer.py emit \
  --members m1d,m1m,m3q32A,m0s,s3s,m2qS2 --split test --out preds_test_rebuilt.jsonl

# 3. score the two labelled splits
python eval/local_scorer.py score data/dev.jsonl   preds_dev.jsonl --convention present,0
python eval/local_scorer.py score data/train.jsonl preds_oof.jsonl --convention present,0

# 4. package a submission
python generate_submission.py --preds preds_test_rebuilt.jsonl --split test \
  --out submission_test_rebuilt.zip
```

The step 2 output names avoid overwriting the archived artefacts. A fresh emit reproduces the
ST1 and ST2 fields of the archived `preds_test.jsonl` exactly and differs from it on the ST3
field of 24 of the 503 rows, because that file was built with an alternative ST3 stacker
configuration.

Refitting the decision layer from the stored member probabilities (`decision_layer.py fit`)
rewrites `decide/params.json`; the archived file is the canonical one and is pinned by
`CONFIG-MANIFEST.json`.

Two notes on the repository. `mirror/` is a clone of a third-party public repository, excluded
from version control and pinned by remote and commit in `UPSTREAM.md`; nothing in the pipeline
reads from it. `preds/` contains more than the six members that entered the blend: per-seed encoder files
(`m1d-s42` to `m1d-s45`, `m1m-s42` to `m1m-s44`), the stacker inputs `m0`, `m3q32B` and
`m3q32C`, the untrimmed `m2q`, and further zero-shot and stacker variants (`m3q32D`,
`m3q32THC`, `s1s`, `s2s`, `s3g`, `s3sX`). The authoritative membership is the `members` list in
`decide/params.json`.


---

## Note on this public release

Instance identifiers and channel names have been replaced with opaque labels (`instance A`..`instance D`). Competition Terms item 5 forbids redistributing the dataset, and item 4 asks that findings not single out an individual creator. Advertiser names are retained because they are companies rather than creators and the argument does not work without them. The dataset itself, the model probability files and the fold assignment are not included here for the same reason; obtain the data from the task organisers.
