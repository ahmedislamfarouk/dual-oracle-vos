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

## Amendment 1 (2026-11-21, before any R1, R3 or Modal run)
Kaggle T4 sessions queue one at a time, so R1, R3 and a second DAVIS check run on Modal NVIDIA A10G GPUs
(SAM 2 commit 2b90b9f, torch 2.7.1, transformers 4.49.0). R2 and the first DAVIS check stay on Kaggle T4.
Hypotheses, populations, arms, thresholds and statistics are unchanged. Each hardware platform gets its own DAVIS check.

## Amendment 2 (2026-11-22, before any analysis of R1 data)
Modal stopped at its spend limit after 39 R1 videos of one arm. Those 39 are not used. Remaining compute is the Kaggle T4 quota, so:
- R1 runs on the first 120 videos of `prereg/mose_fresh240.txt` (list order, fixed now, before any R1 result is analysed).
  All five arms run on Kaggle T4 under float16 autocast. A bfloat16 and float16 run of four DAVIS videos on T4 gave the same
  J&F to three decimals, and float16 is 2.4 times faster on T4. H1 to H4, the margin and the statistics are unchanged.
- R2: the 120-video LVOS run reached its time cap after 53 videos (frame-0 and QRD, bfloat16). H6 is tested on those 53.
  H5 is tested on the same 53 videos, with the three-seed oracle run under float16.
- R3 (Mask R-CNN) is dropped for lack of compute.
- R4: the T4 DAVIS check covers 26 of 30 videos (time cap). It is reported as is.

## Amendment 3 (2026-11-22, before any R2 run on these videos)
A local NVIDIA RTX 4070 Laptop GPU (bfloat16 autocast, same code) runs the 67 LVOS training videos the Kaggle run did not reach
(`prereg/lvos_rest67.txt`): frame-0, QRD and the three-seed oracle. H5 and H6 are then tested on all 120 videos as planned,
and also reported separately for the 53 Kaggle videos and the 67 local videos.
