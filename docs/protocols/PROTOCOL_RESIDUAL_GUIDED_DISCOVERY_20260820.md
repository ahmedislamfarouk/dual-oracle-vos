# Residual-Guided Temporal Discovery (RGTD): frozen protocol

Status: preregistered before the first RGTD result.  This document is a
development and evaluation contract, not a manuscript claim.

## Question

Can an offline, label-free schedule choose three re-detection frames more
effectively than the existing fixed quartile schedule, without increasing the
number of detector calls?

## Method fixed before evaluation

1. Run the unchanged frame-0 DETR + SAM2 candidate pool and propagate its
   tracks over the video.
2. For every later frame, form the union of these propagated masks.  Compute
   the mean absolute grayscale difference from the preceding frame only over
   pixels outside that union.  This is the coverage-residual motion score.
3. Divide the non-reference frames into three equal temporal bins; choose the
   highest-scoring frame in each bin.  Ties choose the earliest frame.  Thus
   every valid video uses exactly three re-detection frames, matching the fixed
   quartile control.
4. At the chosen frames, use the unchanged DETR objectness threshold (0.05)
   and coverage gate (at least 70% of box area outside baseline-track masks).
   Propagate admitted boxes with the existing SAM2 evaluator.  No GT, category
   label, training, or test-time parameter fitting is used by the schedule.

The method is offline because it scores future frames before selecting the
three calls.  This is consistent with the existing multi-frame and backward
propagation setting; it must not be presented as an online method.

## Controls and measurements

Primary evaluation: all 507 YouTube-VOS 2019 validation videos.

Each video is scored by the same evaluator under:

- A: frame-0-only candidate pool;
- B: fixed quartile re-detection at {n/4, n/2, 3n/4}; and
- C: RGTD, three residual-selected frames.

All three use identical model weights, resolution, detector threshold,
coverage gate, metrics, and candidate propagation.  Report greedy and
bijective J&F, per-object recovery, detector calls, admitted tracks, total
runtime, and paired Wilcoxon tests.  Keep DAVIS as a negative control because
its reference-frame protocol has no late-appearing objects.

The initial smoke uses four alphabetically first YouTube-VOS validation videos
only to establish execution and output schema.  It cannot select parameters or
be reported as an efficacy result.  The first full validation run is final for
this fixed algorithm.  If it fails to improve over B at the same three detector
calls, the method is not added to the manuscript as a positive result.

## Prohibited post-hoc changes

Do not change the number of selected frames, temporal-bin rule, residual
formula, detector threshold, coverage threshold, backbone, resolution, or
evaluation protocol after observing validation scores.  A genuinely new method
requires a new dated protocol and a new independent evaluation.

## Implementation amendment — 2026-08-20, before accepting the aggregate

The initial full-run audit found that the integer bin construction selected only
two distinct frames for two videos with fewer than six frames, even though both
had three available non-reference frames.  The residual formula, detector,
thresholds, gates, and all normal-length-video selections are unchanged.  The
implementation now directly splits non-reference frame indices into three bins
only when `n < 6`; it therefore selects three calls whenever three candidates
exist.  The two affected residual videos are rerun once under this correction,
with the original raw output retained.  This is a deterministic execution bug
fix, not parameter selection or a response to their scores.
