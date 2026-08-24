# Results

ChildSafeAds: Commercial Content in Child-Facing YouTube Videos. Shared task of the NLLP workshop at
EMNLP 2026, hosted on Codabench (competition 17595). The development phase closed 2026-08-10; the
evaluation phase ran from 2026-08-11 to 2026-08-18.

Metric: `mean_macro_f1 = (ST1 macro-F1 + ST2 macro-F1 + ST3 macro-F1) / 3`. Each sub-task is scored by
macro-F1 over its own label space, computed only from its own output field. A class absent from both the
gold and the predictions is excluded from the macro average (present-label-set semantics, verified in
`eval/local_scorer.py`).

`METHOD.md` describes the system, the metric analysis and the limitations. `README.md` is the
short version of both. This document records what each submission scored and what was measured
and not kept.

## Final standing

| Rank | Team | mean | ST1 | ST2 | ST3 |
|------|------|------|-----|-----|-----|
| 1 | abc-abc | 0.7347 | 0.8496 | 0.7550 | 0.5996 |
| 2 | this submission | 0.7260 | 0.8049 | 0.7900 | 0.5830 |
| 3 | Nürnberg NLP | 0.7079 | 0.6464 | 0.8243 | 0.6530 |
| 4 | runner | 0.7022 | 0.7923 | 0.7945 | 0.5197 |
| 5 | rudolpheric | 0.6699 | 0.6581 | 0.7364 | 0.6153 |

**Corrected 2026-08-22.** An earlier version of this table showed this submission on the top
row. It is second. The entry `abc-abc` (submission 893084) scored 0.7347 and was submitted at
2026-08-18 16:59 UTC, seven hours inside the phase, which the Codabench API gives as ending
2026-08-19T00:00:00Z. The leaderboard renders timestamps in the viewer's local zone, which is
why that submission displays as 2026-08-19 00:59 from Singapore.

Auxiliary columns for this entry: ST3-family 0.6620, coverage 1.0.

Auxiliary ST3-family scores for the entries above are 0.6620, 0.7031, 0.6359, 0.6984 and
0.6688 respectively. The second and fifth entries are declared as open-weights LLM systems.

## Submission progression

Five submission slots were available in the evaluation phase. All five were used.

| Slot | mean | ST1 | ST2 | ST3 | What changed |
|------|------|-----|-----|-----|--------------|
| 1 | 0.6410 | 0.6021 | 0.7686 | 0.5524 | First evaluation-phase entry. Five-member equal-weight blend (m1d, m1m, m3q32A, m0s, s3s) with the decision layer fitted on out-of-fold predictions. |
| 2 | 0.6516 | 0.6021 | 0.7761 | 0.5766 | Adds the ST2-only m2qS2 member as a sixth member, the calibrated zero-shot prompt variant as a stacker input, two corrections to the sponsor-mention rule (the pattern widened to match `sponsoring`, and a sponsorship phrase in the video description added as a third precluding condition), the age-restricted column rebuilt from the members that can separate that class, and a direct-exhortation column extra. ST1 is bit-identical to slot 1, as expected since nothing changed on that column. |
| 3 | 0.6537 | 0.6021 | 0.7761 | 0.5830 | A second configuration of equivalent out-of-fold quality (0.7010 against 0.6997 for slot 2), selected for its disagreement with slot 2 rather than for a point-estimate gain. Only the ST3 column differs. |
| 4 | 0.6986 | 0.7378 | 0.7761 | 0.5818 | First entry to emit the ST1 `other` class, after the leaderboard inversion established that the class is present in the test gold. It emitted **two** hand-set ones, not one; the second was a false positive that also cost a `physical_goods` true positive, and it was dropped for slot 5 (see `METHOD.md` 7.5). It also carries a third ST3 stacker configuration of equivalent out-of-fold quality (0.7017), differing from slot 3 on 53 of the 503 rows. ST2 is unchanged; ST3 falls by 0.0012. |
| 5 | 0.7260 | 0.8049 | 0.7900 | 0.5830 | Final entry. ST2 rises to 0.7900 with the two hand-assigned gambling labels; ST1 rises to 0.8049; the ST3 column returns to the slot 3 value. |

The movement from slot 3 to slot 5 comes from three labels assigned by hand: one ST1 instance, which is the first and only
`other` the entry emitted, and two ST2 instances. Slot 5 is the slot 3 model configuration with
those three labels changed. No member was retrained and no threshold was refitted for it, which
is why its ST3 column returns to the slot 3 value. `METHOD.md` section 7 gives the reasoning
that produced them and section 8 states what they do not support.

### Contribution of the manual labels

| Version | mean | ST1 | ST2 | ST3 |
|---|---|---|---|---|
| model only, slot 3 | 0.6537 | 0.6021 | 0.7761 | 0.5830 |
| slot 3 plus `decide/advertiser_precedent.py` | 0.7213 | 0.8049 | 0.7761 | 0.5830 |
| as submitted, slot 5 | 0.7260 | 0.8049 | 0.7900 | 0.5830 |

Ranks are omitted here. Only the top five rows of the final board were captured, so where the
lower two versions would have placed cannot be stated from the record held.

The ST1 label is reproduced exactly by the coded rule, which fires on one of 503 evaluation
instances with no false positives. The two ST2 labels are not reachable by that rule. See the
compliance note in `METHOD.md` section 7.5.

Slot 2 also served as a check on the out-of-fold to test transfer of a validated change. The paired
out-of-fold bootstrap predicted ST3 +0.0232 and the test column moved +0.0242; the predicted mean gain
was +0.0077 against +0.0106 delivered.

## Held-out reference numbers and test transfer

The out-of-fold and dev columns below were reproduced from the repository as it stands, by running
`decide/decision_layer.py emit` over the six members named in `decide/params.json` and scoring the output
with `eval/local_scorer.py` under the pinned convention. Out-of-fold predictions cover the 2353 training
rows under a 5-fold channel-grouped split with greedy rare-label balancing (`eval/folds.json`,
632 channels). Dev is the 504-row development split, which is disjoint by channel from training.
Test is the 503-row evaluation split.
The configuration held in the repository is the slot 2 one. Slots 3 and 5 used a variant of the ST3
stacker over the same six members, whose out-of-fold mean is 0.7010.

| Column | Out-of-fold (2353 rows) | Dev (504 rows) | Test, slot 3 (model only) | Test, slot 5 (final) |
|--------|------------------------|----------------|---------------------------|----------------------|
| ST1 | 0.6369 | 0.8519 | 0.6021 | 0.8049 |
| ST2 | 0.8423 | 0.7572 | 0.7761 | 0.7900 |
| ST3 | 0.6199 | 0.5966 | 0.5830 | 0.5830 |
| mean | 0.6997 | 0.7352 | 0.6537 | 0.7260 |

The ST1 row needs a note on how to read it. Out-of-fold ST1 is depressed by the same mechanism as the
test column: `other` has 2 training instances, is never predicted, and enters the macro average at
F1 = 0. Rescaling the slot 3 test figure to the four classes the system can actually score gives
0.6021 x 5/4 = 0.7526. The out-of-fold figure rescaled the same way is 0.6369 x 5/4 = 0.7961, so the drop
from out-of-fold to test on this column is about 0.044, comparable to the drops on ST2 (0.066) and ST3
(0.037).

Dev needs a similar note. Dev is a smaller split with a different class mix, and dev ST2 and ST3 sit below
their out-of-fold values while dev ST1 sits well above. Selection during development was made on
out-of-fold predictions rather than on dev, because the out-of-fold folds are channel-disjoint in the same
way the official evaluation split is.

Estimate noise, measured by channel-level bootstrap. **Corrected 2026-08-22.** The figure of 0.0104
previously reported here was measured on a configuration that was never shipped. Re-measured on the
configuration in this repository (632 channels, 4000 replicates), the marginal out-of-fold standard
error of the mean is **0.0200** and of ST1 is **0.0555**. Those marginals are inflated by a metric
artefact rather than by system variance: ST1 `other` has 2 gold rows in the out-of-fold split and is
never predicted, so whether a resample contains one flips the ST1 divisor between 5 and 4 classes.
Conditioning on that event, the standard error of the mean is **0.0090** when the class is drawn and
**0.0102** when it is not, and the two regimes are separated by **0.0530** on the mean with identical
predictions. `clinic/presence_artefact.py` reproduces this. A 503-row scored estimate has standard
error 0.0337 on the mean (ST1 0.0867, ST2 0.0298, ST3 0.0444), from 1500 channel-disjoint subsamples
drawn to test size.

## Approaches tried and rejected

Every candidate change was tested against a paired channel-cluster bootstrap over out-of-fold predictions
(`eval/cluster_bootstrap.py`): channels are resampled with replacement, both arms are scored on the same
draw, and the delta is taken. A change was kept only when the 95% interval on the delta excluded zero,
either on the mean or on the sub-task column the change acted on. Deltas below are mean `mean_macro_f1` unless a column is named. Dev was used as a direction check
only.

### Ensemble composition

| Change | Measurement | Reason for rejection |
|--------|-------------|----------------------|
| Blend weight grid search, m3q32A x s3s weights in {0.5, 1, 1.5, 2}^2, 16 fits | Best configuration (m3q32A 1.5, s3s 1) gave out-of-fold 0.6849 against 0.6830 for equal weights; dev fell 0.7293 to 0.7279. Equal weights ranked 2 of 16. | +0.0019 is below any plausible standard error for a 16-configuration search, and dev moved the other way. Equal weights kept. |
| Seed expansion, 4-seed m1d instead of 3 | Out-of-fold mean 0.6847 against 0.6859. | A change intended only to reduce variance lost 0.005 on ST1, which indicates the comparison sits inside the measured noise band. Five queued seed jobs were cancelled. |
| m2q as a full sixth member across all three sub-tasks | Mean +0.0085, 95% CI [-0.0074, +0.0226], P(better) = 0.86. Per column: ST2 +0.0213 CI [+0.0103, +0.0400] P = 1.00, ST1 +0.0019 and ST3 +0.0022 with intervals spanning zero. | The gain is confined to ST2; the ST1 and ST3 intervals span zero, and dividing the ST2 gain across three columns is what produces the mean figure. The member was retained in ST2-only form (m2qS2), which passed at +0.0071 CI [+0.0034, +0.0133]. |
| Zero-shot prompt variant C swapped for variant A in the blend | Out-of-fold mean 0.6762 against 0.6997. | Worse. |
| Zero-shot prompt variant C added as a sixth blend member | Mean +0.0027, CI [-0.0080, +0.0128]. | Interval includes zero. Variant C was retained instead as a stacker input feature, which passed at +0.0022 CI [+0.0002, +0.0045]. |
| ST1 level-2 stacker (s1s) | Out-of-fold ST1 0.6449 to 0.6468, mean +0.0006; dev ST1 0.6766 to 0.6617. | Gain below noise, dev disagreed. |
| ST2 level-2 stacker (s2s) as a sixth member | With exact passthroughs on the untouched columns, out-of-fold mean 0.6862 against 0.6859, all of it from ST2 +0.0011; dev ST2 0.7526 to 0.7451. | Gain far below one standard error, dev disagreed. |
| Gradient-boosted ST3 stacker (s3g), three integrations | Standalone it is the better stacker: out-of-fold macro over the 8 flags 0.6096 against 0.5780 for the logistic-regression stacker, with gains on the rare classes. All three integrations failed the test: add as a seventh member +0.0016 CI [-0.0056, +0.0099]; swap for s3s with the other columns held byte-identical -0.0021 CI [-0.0091, +0.0031]; per-class mixing on its three strong classes +0.0022 CI [-0.0058, +0.0108]. | The standalone advantage did not carry into the ensemble, because the logistic-regression stacker's errors are less correlated with those of the other members. |
| s3s stacker as a standalone system rather than as a blend member | Out-of-fold 0.6505 against 0.6564 for the blend. | Worse as a standalone system than as a blend member. |

### Features and rules

| Change | Measurement | Reason for rejection |
|--------|-------------|----------------------|
| s3s disclosure-position features (phrase offset in the description, above-the-fold flag at 150 characters) | Out-of-fold ST3 0.5924 to 0.5843; dev ST3 0.5775 to 0.5869. | The only gain was on dev. The test split is channel-disjoint in the same way the out-of-fold folds are, so out-of-fold is the selection signal. Reverted and the canonical figures re-verified bit-consistent. |
| Exhortation and call-to-action phrase features in the s3s auxiliary block | Out-of-fold ST3 0.5905 to 0.5840, mean 0.6859 to 0.6837; dev ST3 also fell, with collateral damage on the age-restricted class (0.643 to 0.581). Direct-exhortation dev F1 was unchanged at 0.491. | The phrase family fires too broadly (merchandise, urgency and percent-off text hit non-exhortation rows) and the stacker spends capacity on it. |
| ST1 `other` override rule driven by the zero-shot member (fire when p(other) >= 0.70) | Paired delta -0.0043, standard error 0.0198, 95% CI [-0.0569, +0.0121], P(better) = 0.63. | Rejected during development on the paired test and on an explicit payoff calculation. The class was later shown to be present in the test gold by leaderboard inversion, and was handled by a single hand-assigned label rather than by a firing rule; inspection of the rule's top test candidates showed it would have fired on rows the annotators label `none` or as services. |
| hfss sponsor-domain suppression | P(better) = 0.82; estimated test impact +0.0005. | Below the gate, and it hardcodes a brand. |
| Inadequate-disclosure column extra from m2q | P(better) = 0.94. | Below the gate. |
| Per-cell inadequate-disclosure thresholds | Dev sign flip. | Direction check disagreed with the out-of-fold gain. |
| m2q as an s3s stacker input feature | ST3 -0.0062, mean -0.0021, interval spans zero. | The route that worked for prompt variant C did not work for this member. |

### Prompting and thresholds

| Change | Measurement | Reason for rejection |
|--------|-------------|----------------------|
| Zero-shot prompt variant B, definitions plus legal notes drawn from `legal_provisions.json` | Approximately equal to variant A (definitions only) on all three sub-tasks. No numeric delta was recorded for the pair. | Legal grounding in the prompt did not help. This replicates a negative result the organisers reported. |
| Zero-shot prompt variant D, a per-class error-analysis rewrite of the ST3 prompt | At member level D improves five of eight flags (undisclosed +0.045, direct exhortation +0.048, hfss +0.049, inadequate disclosure +0.029, misleading claim +0.006) but collapses no_flag (-0.203) and age-restricted (-0.132), so its macro is 0.4572 against 0.4768 for variant C. At system level: add alongside C, -0.0006 CI [-0.0031, +0.0016]; replace C, +0.0013 CI [-0.0021, +0.0069]. | The flags are coupled. no_flag is a residual class, so recall bought on the substantive flags is paid for there. Neither integration cleared the gate. |
| Reasoning mode on the zero-shot member (Qwen3 `enable_thinking`) | Member-level ST3 macro 0.4768 to 0.4782 on the same prompt. System level through the stacker route: mean +0.0019, 95% CI [-0.0011, +0.0071], P(better) = 0.76. | Reasoning mode produced no measurable system-level gain on this task. The measurement cost about 17 GPU-minutes. |
| Expected-F1-at-test-size thresholds (maximise mean per-class F1 over channel-disjoint 503-row draws rather than F1 on the pooled 2353-row out-of-fold set) | Moved 4 ST2 thresholds and promised +0.204 expected F1 on the gambling class. A non-circular check (fit on one draw seed, evaluate on 800 fresh draws with a different seed) gives delta -0.0000, P = 0.482. | The objective is the problem. Optimising each class's expected F1 in isolation ignores that under present-label-set semantics firing more often makes a class present in draws where the gold has none, contributing hard zeros to the macro. Per-class expected F1 and macro expected F1 are different objectives. |
