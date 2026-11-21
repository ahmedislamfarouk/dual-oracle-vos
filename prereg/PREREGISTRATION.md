# Pre-registered follow-up runs

Written 2026-11-21 before any of these runs. Committed publicly before the first run starts.
Code, thresholds and evaluator are frozen at the `dovos` package of this repository. Nothing is tuned.
Hardware: Kaggle NVIDIA T4 GPUs, SAM 2.1 Hiera-Large under bfloat16 autocast, DETR-R50 objectness floor 0.05,
admission threshold 0.5. A DAVIS-2017 val re-run (frame-0 and QRD) checks agreement with the archived A6000 runs
(82.73 and 86.71) before the new populations are read.

## R1. Fresh capped MOSE (post-hoc margin, sample size)
- Population: MOSE training videos at positions 121 to 360 of the identifier-sorted `meta_train.json`
  (240 videos, none used before), first 60 annotated frames. List: `prereg/mose_fresh240.txt`.
- Arms: frame-0, QRD, O1, O4 one seed (best frame), O4 three seeds (best frames).
- Confirmatory hypotheses, Holm-corrected over the five:
  - H1: QRD above frame-0.
  - H2: O1 and QRD equivalent at a margin of +/-4 pp, meaning the 90% bootstrap interval of O1 minus QRD lies inside +/-4.
  - H3: O4 one seed and QRD equivalent at +/-4 pp, same rule.
  - H4: O4 three seeds above QRD.
- Significance: two-sided Wilcoxon signed-rank p < 0.05 and a 95% paired percentile bootstrap interval
  (20,000 resamples, seed 0) excluding zero. A video missing from any arm is dropped from that comparison.

## R2. Clean holdout (SAM2 training overlap, LVOS power)
- Population: the 120 LVOS-v1 training videos (not in the SAM 2 training mixture, never used by this paper),
  100 annotated frames spread over each clip, as in the paper's LVOS run.
- Arms: frame-0, QRD, O4 three seeds.
- H5 (confirmatory): O4 three seeds above QRD. H6 (confirmatory): QRD above frame-0.

## R3. Second detector (one pipeline family)
- torchvision Mask R-CNN R50-FPN (COCO, DEFAULT weights), every box with score >= 0.05 kept regardless of class,
  ordered by score, in place of DETR. Everything else unchanged. Population: the 118 capped MOSE videos of the paper.
- Arms: frame-0, QRD, O4 one seed. Exploratory replication of H1 and H3.

## R4. Cost
- DAVIS-2017 val 30, frame-0 and QRD, wall-clock per video and detector calls on one T4. Descriptive only.

## Reporting rule
Every result is reported whatever its sign. Confirmatory results go in the main text.
If the DAVIS check differs from the archive by more than 1 pp in either arm, that is reported next to every new result.
