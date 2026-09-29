# CPD-2 adaptive discovery: pre-implementation protocol

**Status:** frozen design specification. The five-video DAVIS engineering smoke
passed its implementation/integrity gate on 2026-08-24; no confirmatory CPD-2
result exists yet.

## Research question

Under the completed CPD-1 matched interface, can a label-free offline temporal
discovery policy improve a fixed three-later-call detector budget over fixed,
seeded-random, and global-motion schedules, and does its benefit vary with the
stage diagnosed as limiting?

This is the required methodological extension if an IJCV paper is pursued. It
must not be conflated with the prior YouTube-VOS-only RGTD evidence, which used
a different evaluator/implementation.

## Populations and split discipline

- Development: DAVIS-2017 validation only, used solely to complete engineering
  parity gates and select no numerical method parameter.
- Confirmatory populations: full YouTube-VOS 2019 validation (507 videos) and
  full LVOS-v1 validation (50 videos). The CPD-2 policy and all constants must
  be frozen before either confirmatory run.
- Do not tune on either confirmatory result. Report every arm and every
  completed population.

## Fixed common interface

- 854x480, SAM2.1 Hiera-Large, DETR first-frame and later-frame proposer,
  top-10 per detector call, objectness floor 0.05, the CPD-1 commitment rule,
  strict committed-set Hungarian score, and query-visible A/B/C audit.
- Base arm: first-frame proposals/propagation only.
- Every temporal arm receives exactly three *later* DETR calls. New boxes are
  admitted only when at least 70% of their box area is uncovered by the fixed
  baseline propagated-mask union. The same baseline union is used by the
  residual schedule; later discoveries do not update it. New tracks propagate
  forward from their discovery frame; pre-discovery frames are empty. No
  annotation, category label, or GT-derived schedule is permitted.
- Scheduling is offline and must be labelled as such.

## Pre-specified arms

1. **Fixed:** `{n/4, n/2, 3n/4}` (deduplicate invalid indices).
2. **Seeded random:** one uniform frame per temporal third, seed 20260824.
3. **Global motion:** greatest grayscale frame-to-frame change per temporal
   third over all pixels (earliest tie).
4. **Residual guided:** greatest grayscale change outside current propagated
   mask union per temporal third (earliest tie).

Every arm has the identical later-detector-call budget. Detector, thresholds,
admission rule, propagator, and evaluator must not vary by arm.

## Required engineering/evidence gates

1. Implement CPD-2 as a new driver; do not adapt the incomplete
   `ijcv_revision` temporal-discovery module or mix results from the prior
   YTVOS-only evaluator.
2. On a fixed DAVIS smoke manifest, show all schedule arms are deterministic,
   use the required call counts, never read GT before inference, and have
   restart-safe per-video output.
3. Establish CPU-pair-parallel and GPU-resident-state parity against the frozen
   CPD-1 output before using either runtime change. A future GPU score-matrix
   backend must likewise match discrete decisions and numeric outputs exactly.
4. Run fixed/global/residual/random on the complete confirmatory populations;
   use paired video-level contrasts, 20,000 fixed-seed bootstraps, and strict
   output merging/checksums.

## Decision rule

An IJCV revision is supported only if the completed confirmatory evidence shows
a useful, uncertainty-resolved adaptive-policy advantage or another equally
substantive, protocol-valid method finding. If effects are null/mixed, retain
the TMLR manuscript and release CPD-1 as a careful diagnostic follow-on rather
than overstate novelty.
