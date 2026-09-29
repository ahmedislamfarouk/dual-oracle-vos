# RGTD global-motion mechanism ablation: frozen follow-up protocol

Status: frozen before the first global-motion result. This is a follow-up
mechanism ablation after the completed RGTD validation; it is not part of the
original preregistered fixed-versus-residual experiment.

## Question

Does RGTD's use of *uncovered* temporal change contribute beyond selecting the
largest temporal change anywhere in the frame, at the same detector-call budget?

## Frozen arms

All arms use the same frozen DETR, SAM2 Large, 854x480 input resolution,
objectness floor 0.05, 70% uncovered-area admission gate, forward propagation,
strict bijective evaluator, and exactly three later detector calls. The original
full 507-video outputs for the first and third rows are reused only as already
completed reference arms; this new run executes the middle row.

| Arm | Selection rule | Evidence state |
|---|---|---|
| Fixed | Calendar times `{n/4,n/2,3n/4}` | completed 2026-08-20 |
| Global motion | Maximum grayscale inter-frame difference over **all** pixels in each of three equal temporal bins | execute now |
| RGTD | Maximum grayscale inter-frame difference outside baseline propagated-mask union in each bin | completed 2026-08-20 |

Ties select the earliest frame. For videos with fewer than six frames, direct
splits of the available non-reference indices ensure three distinct calls where
three candidates exist. This is the same execution correction applied to the
accepted RGTD aggregate.

## Evaluation

Primary population: all 507 YouTube-VOS 2019 validation videos, partitioned
deterministically into four shards by sorted video ID modulo four. Primary metric:
macro strict-bijective J&F. Primary contrast: RGTD minus global motion. Secondary
contrasts: global motion minus fixed, track count, new-track count, and late-object
recovery. Compute one-sided paired Wilcoxon tests for RGTD > global and a
deterministic 20,000-resample percentile bootstrap interval for the macro mean
difference.

The global score is intentionally parameter-free. No threshold, frame budget,
temporal-bin rule, detector setting, gate, backbone, or dataset subset may change
after observing output. All raw logs/JSONs must be copied locally and checksummed
before a manuscript claim is added.

## Interpretation contract

If RGTD exceeds global motion, it supports the narrow claim that excluding
already-covered pixels helps choose three re-detection times in this frozen
offline DETR+SAM2 setup. If no difference is resolved, retain RGTD as a
matched-budget scheduling policy but do not attribute its effect specifically to
the coverage residual. A negative result is reported, not discarded.
