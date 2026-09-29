# A Dual-Oracle Protocol for Annotation-Free Video Object Segmentation

Code and per-video results for **"A Dual-Oracle Protocol for Annotation-Free Video Object Segmentation:
Separating Discovery Quality from Discovery Timing"** by Ahmed Islam, Ammar Toutou, Youssef Rezk,
Belal Fathy and Abdallah Zain (Alamein International University).

An annotation-free video object segmentation pipeline can be built from frozen public models: a
class-agnostic detector proposes objects on a reference frame and a promptable segmenter propagates
them. Such a pipeline loses accuracy when the detector is wrong or when it is asked at the wrong moment,
and an ordinary ablation cannot separate the two. This repository implements a **dual-oracle protocol**
that does:

- **O1, detection oracle:** ground-truth boxes at the reference frame, real timing.
- **O4, timing oracle:** the real detector, with each object seeded at the frame where the detector's box
  overlaps it best.

Both oracles use the pipeline's own propagation and ground-truth-free commitment, so they are reference
points for one failure mode each rather than formal upper bounds.

## Pipeline

| Stage | Component |
|---|---|
| Proposals | DETR-R50, class-agnostic objectness `1 - p(no-object) > 0.05` |
| Propagation | SAM 2.1 Hiera-Large, bfloat16 autocast |
| Re-detection (QRD) | detector re-run at frames `n/4`, `n/2`, `3n/4`; a box is admitted when existing tracks cover less than half of it; bidirectional propagation |
| Commitment | ground-truth-free: persistence >= 0.5, mean area < 0.5, temporal-IoU de-duplication at 0.5 |

Nothing is trained or tuned per benchmark.

## Main results

Committed-track bijective J&F (%), our evaluator. Every comparison is paired over identical videos.

| Arm | capped MOSE (n=118) | DAVIS-2017 (n=30) | YouTube-VOS, late objects (n=89) | LVOS (n=50) |
|---|---|---|---|---|
| Frame-0 baseline | 51.03 | 82.73 | 42.09 | 67.81 |
| QRD (four fixed detector calls) | 63.96 | 86.71 | 55.31 | 74.80 |
| O1: ground-truth boxes at frame 0 | 63.87 | 86.69 | | |
| O4: best frame, one seed | 65.06 | 87.38 | 57.68 | 73.21 |
| O4: best frames, three seeds | 68.19 | | 60.93 | 77.00 |

No single-seed oracle is resolved above QRD. Three best-frame seeds are resolved above it on capped MOSE
(+4.23) and on late-appearing YouTube-VOS objects (+5.61), but not on LVOS (+2.20). The full set of 107
tests, with confidence intervals, is listed in the paper's supplementary material.

## Repository layout

| Path | Contents |
|---|---|
| `mcvos/` | Pipeline, DETR objectness detector wrapper and J&F evaluator |
| `experiments/evaluation/` | Benchmark evaluation (DAVIS-2017/2016, YouTube-VOS, MOSE, LVOS, SegTrack v2) and ground-truth-free commitment |
| `experiments/qrd_and_schedules/` | QRD, forward-only and dense schedules, coverage-gap re-detection, recovery coefficient |
| `experiments/oracles/` | Dual-oracle arms, frame choice x seed count, non-overlapping and top-20 re-scoring; `x3_jobs.txt` lists every configuration run |
| `experiments/baselines_and_controls/` | Detector-architecture, proposal-source, backbone, propagator and external-baseline controls |
| `analysis/` | Paired bootstrap and Wilcoxon analyses that produce the numbers in the paper |
| `results/` | Per-video results (`results.json` / `.json` / `.csv`) for every reported arm |
| `docs/` | Experiment protocols and the DAVIS full-sequence reproduction guide |

## Installation

```bash
git clone https://github.com/ahmedislamfarouk/dual-oracle-vos && cd dual-oracle-vos
pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam2 && pip install -e sam2
(cd sam2/checkpoints && ./download_ckpts.sh)
export PYTHONPATH=$PWD/mcvos:$PYTHONPATH
```

## Data

The benchmarks are public and are not redistributed here: DAVIS-2017 and DAVIS-2016, YouTube-VOS 2019,
MOSE, LVOS v1 and SegTrack v2. Each experiment reads a dataset root laid out as
`<ROOT>/JPEGImages/<video>/*.jpg` and `<ROOT>/Annotations/<video>/*.png`, passed through the `ROOT`
environment variable. Capped MOSE is the first 60 frames of the first 120 MOSE training videos in
identifier order.

## Reproducing

Re-run the statistics from the archived per-video results (no GPU needed):

```bash
python analysis/analyze_x3.py      # dual-oracle, frame choice x seed count, non-overlapping re-scores
python analysis/analyze_x2.py      # admission sweep and multi-seed oracle
python analysis/label_budget.py    # how many labelled videos the deployment decision needs
```

Re-run an arm from scratch (one GPU), for example QRD and the timing oracle on capped MOSE:

```bash
ROOT=/data/mose_subset/train MAXF=60 TAG=qrd_mose python experiments/oracles/tide_x3.py
ROOT=/data/mose_subset/train MAXF=60 ORACLE=4 OSEEDS=3 OFRAMES=best TAG=o4_mose \
    python experiments/oracles/c4_x3.py
```

`experiments/oracles/x3_jobs.txt` gives the exact environment of every run reported in the paper.
Both scripts write `results/<TAG>/results.json` and find SAM 2 in `./sam2`; override with `SAM2_DIR`,
`CKPT_DIR` and `RESULTS_DIR`. Scripts in `experiments/evaluation/`, `qrd_and_schedules/` and
`baselines_and_controls/` still refer to the original project root as `/path/to/ow_vis`, a directory
holding `src/` (this repository's `mcvos/`), `sam2/`, `checkpoints/` and `results/`; set that path before
running them.

## Citation

```bibtex
@misc{islam2026dualoracle,
  title  = {A Dual-Oracle Protocol for Annotation-Free Video Object Segmentation:
            Separating Discovery Quality from Discovery Timing},
  author = {Islam, Ahmed and Toutou, Ammar and Rezk, Youssef and Fathy, Belal and Zain, Abdallah},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

MIT; see `LICENSE`. DETR and SAM 2 are used under their own licenses.
