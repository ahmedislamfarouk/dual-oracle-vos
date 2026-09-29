# CPD-1: Cross-Proposer Discovery Decomposition

**Status:** frozen before implementation smoke or any new efficacy result (2026-08-21).

## Question

Under one propagated-proposal budget and a common downstream pipeline, does the
largest observed loss remain at discovery across materially different frozen
proposal families? This is a bounded empirical question; it does not establish
that every annotation-free VOS system has the same bottleneck.

## Populations and roles

| Population | Role | Scope |
|---|---|---|
| YouTube-VOS 2019 validation | Primary | all 507 validation videos, complete annotated sequences |
| LVOS v1 validation | Independent confirmation | all available official validation sequences, complete annotated sequences |
| DAVIS-2017 validation | Negative/control population | all 30 validation sequences, complete sequences |
| MOSE training split | Development only | never called external validation and never used to tune CPD-1 post hoc |

**Gate 0:** do not launch the reported LVOS arm unless the complete validation
frames and annotations are verified on the server. If absent, record that fact
and retain CPD-1 as a two-dataset study; do not make the planned cross-dataset
generality claim.

## Frozen proposal families

1. **DETR:** the archived frozen `facebook/detr-resnet-50` objectness proposer,
   retaining detections with objectness at least 0.05.
2. **Mask R-CNN:** torchvision `maskrcnn_resnet50_fpn` COCO weights; use final
   predicted boxes with labels ignored and score at least 0.05. It is
   COCO-pretrained, not category-agnostic in pretraining.
3. **SAM2 AMG:** the frozen automatic-mask-generator configuration: 16 points
   per side, predicted-IoU threshold 0.80, stability threshold 0.90, box NMS
   0.50, and minimum mask area 400 pixels. Convert every accepted mask to its
   tight box and use only this box to prompt the video propagator. AMG shares a
   model family with the propagator and is not an independent semantic model.

Grounding DINO is not a primary arm. It may be added only as a separately
labelled sensitivity arm with one fixed universal prompt, never dataset GT
category names or validation-tuned prompts.

## Common downstream interface

- Resize every input to 854x480.
- Rank native proposals by their native score and propagate the top 10 per
  queried frame. Retain raw native proposals to compute proposal-only recall at
  K={5,10,20}. Equal top-K controls propagated-track count; it does not equalize
  proposer FLOPs, semantic priors, or total runtime.
- Use unchanged SAM2 Hiera-Large box-prompt propagation, frozen commitment
  thresholds, and the strict bijective Hungarian evaluator.
- No GT image, mask, class, timing, or object-count signal may enter proposal
  generation, scoring, or track commitment.

## Stage definitions and outcomes

For each GT object visible in the actual queried frame, define:

- **A (proposal hit):** at least one retained frame proposal has box IoU >=0.50
  with that object.
- **B (conditional propagation hit):** conditional on A, at least one seeded
  SAM2 track has mean annotated-frame J&F >=0.50 against that object.
- **C (commitment hit):** conditional on B, at least one such track survives the
  frozen commitment rule.

For N objects, report discovery loss `(N-A)/N`, propagation loss `(A-B)/N`, and
commitment loss `(B-C)/N`. These additive losses sum without the non-additive
factorization/Shapley issue removed from the TMLR manuscript. Also report
strict-bijective J&F, proposal and committed object precision/recall, native and
top-K proposal counts, committed-track counts, wall time, peak GPU memory, and
per-video/per-object records.

GT objects absent from the queried frame are **not** spatially comparable to a
frame-0 proposal and are excluded from this A/B/C denominator. Record their
count separately as the structural single-shot blind spot; do not describe it as
a box-recall loss or combine it arithmetically with the queried-frame stages.

## Primary analysis and decision rule

Use 20,000 deterministic video-level bootstrap resamples for every
proposer--dataset cell. Compute discovery loss minus the larger downstream loss.
The intended generality gate passes only if its 95% interval is above zero for
at least two of the three proposal families on both YouTube-VOS and full LVOS.
All results, including failures of this criterion, will be archived and reported
without threshold tuning after validation.

## Execution gates

1. **Schema smoke:** five predetermined videos validate paths, frame lists,
   resolution, model loading, deterministic IDs, exact top-K, GT isolation, and
   evaluator parity. Its efficacy values are non-reportable and cannot change
   the protocol.
2. **Parity audit:** record and compare video IDs, frame lists, SAM2 hash,
   commitment/evaluator hashes, K, and environment for every arm.
3. **Full execution:** run each full population exactly once; preserve raw
   per-object/per-video JSON, logs, model/environment manifests, and SHA-256
   checksums before aggregation.
4. **No silent changes:** no frame caps, video drops, threshold changes, or K
changes after the smoke. OOM/error cases are recorded and addressed only by a
dated protocol amendment before resuming.

### Dated implementation amendment — 2026-08-22: empty retained proposal sets

The original DETR full attempt stopped at its 158th sorted YouTube-VOS video
because the frozen 0.05 threshold produced zero retained proposals and SAM2
rejects an empty prompt set. The original Mask R-CNN attempt encountered the
same executor edge case after its 371st logged video. This is an executor edge
case, not an exclusion: for **every proposer** and every `topk=0` video, do not
instantiate/propagate SAM2; emit empty tracks, empty commitment, and the
ordinary GT-side all-miss evaluation record instead. This does not change the
detector, threshold, frame population, top-K cap, commitment, GT isolation, or
definition of any non-empty-proposal video.

Before restarting an invalid incomplete proposer attempt, a one-video
non-reportable regression smoke must show its exact failure case completes with
`native=0`, `topk=0`, and `committed=0`. Its all-507 result is then a new
complete attempt; every original final-only partial run is preserved as a
failed engineering artifact and is never aggregated.

## Scope limits

This protocol does not test online operation, dense temporal scheduling,
detector-independent causality, or state of the art. Query-visible and
query-absent status are evaluation conventions, not physical object-entry-time
labels.
