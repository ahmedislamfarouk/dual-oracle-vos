# Reproduce the MC-VOS Full-Sequence DAVIS Result

This runbook reproduces the primary committed-track evaluation: DAVIS-2017 validation, all 30 official sequences, full temporal extent, $854\times480$, and strict GT-free track commitment.

## Frozen result to verify

The expected result is stored in `raw_results/davis_fullseq_committed_20260813.json`:

| Measure | Expected value |
|---|---:|
| Sequences | 30 |
| `MAXF` | 9999 (complete sequence extent) |
| Candidate-pool upper-bound J\&F | 83.33 |
| Strict GT-free committed J\&F | 82.73 |
| Retention | 99.27% |

Run `python3 scripts/verify_evidence_bundle.py` from the repository root before relying on the archived result.

## Assemble a clean checkout

Create a working directory, referred to below as `MCVOS_ROOT`, with these contents:

```text
MCVOS_ROOT/
  src/                 # copy reproducibility/canonical_release/src/
  experiments/         # copy reproducibility/canonical_release/experiments/
  checkpoints/
    sam2.1_hiera_large.pt
  sam2/                # official SAM2 source at the pinned commit below
```

`canonical_release/` is an exact source snapshot of the server-side `src/` and `experiments/` directories used for the run. Its full-sequence driver SHA-256 is `97273820e1b6ec69ca53b8517a83cf2bc022592ae59212c630dc0a8c5b5883fe`.

Clone and pin SAM2:

```bash
git clone https://github.com/facebookresearch/sam2.git sam2
git -C sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
# Install CUDA-compatible PyTorch first, then disable build isolation here.
# SAM2's build metadata otherwise resolves a second, newer PyTorch/CUDA stack.
python3 -m pip install --no-build-isolation -e sam2
```

The original run used Python 3.10.12, PyTorch 2.9.0+cu128, Transformers 4.57.3, OpenCV 4.13.0, SciPy 1.15.3, and NumPy 2.2.6. Install a CUDA-compatible PyTorch build first, then use `reproducibility/requirements.runtime.txt` and the pinned SAM2 checkout. Use `--no-cache-dir` for temporary/space-limited environments. `canonical_release/requirements.server.txt` and `canonical_source/pip_freeze_server_20260813.txt` remain the complete server provenance records.

**Space note.** A clean CUDA 12.8 PyTorch environment consumed about 6.8 GB during validation. On the canonical server, `/tmp` is accounted against the user quota, so create the environment on a quota-exempt scratch volume or another machine with at least 10 GB free. Keep package caching disabled (`PIP_NO_CACHE_DIR=1`) for ephemeral smoke tests.

Download SAM2.1 Hiera Large from the official SAM2 release and place it at `checkpoints/sam2.1_hiera_large.pt`. Its required SHA-256 is:

```text
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
```

The detector is loaded by Transformers as `facebook/detr-resnet-50` at revision `no_timm`. The canonical server resolved this to snapshot `70120ba84d68ca1211e007c4fb61d0cd5424be54`; verify the model/config/preprocessor hashes in `reproducibility/DETR_ARTIFACT_MANIFEST.md` after download.

## Dataset

Download DAVIS-2017 and set `DAVIS_ROOT` to its `DAVIS` directory. The runner expects:

```text
$DAVIS_ROOT/ImageSets/2017/val.txt
$DAVIS_ROOT/JPEGImages/480p/<sequence>/*.jpg
$DAVIS_ROOT/Annotations/480p/<sequence>/*.png
```

## Run

From the paper repository root:

```bash
export MCVOS_ROOT=/absolute/path/to/MCVOS_ROOT
export DAVIS_ROOT=/absolute/path/to/DAVIS
export SAM2_CKPT="$MCVOS_ROOT/checkpoints/sam2.1_hiera_large.pt"
export OUT_DIR="$MCVOS_ROOT/results/reproduction_fullseq"

CUDA_VISIBLE_DEVICES=0 MAXF=9999 NSEQ=30 OFFICIAL=1 \
  python3 reproducibility/run_davis_fullseq_portable.py
```

## Mandatory preflight and one-sequence smoke test

Before a full run, validate the filesystem, official sequence list, checkpoint identity, and runtime imports without loading a model:

```bash
python3 scripts/check_reproduction_environment.py \
  --mcvos-root "$MCVOS_ROOT" --davis-root "$DAVIS_ROOT" \
  --sam2-ckpt "$SAM2_CKPT" --out-dir "$OUT_DIR" \
  --verify-checkpoint --check-python --sequences 1
```

Then use a distinct output directory for a one-sequence, one-frame execution smoke test. This checks the evaluator wiring only; it does not validate the headline metric:

```bash
export OUT_DIR="$MCVOS_ROOT/results/reproduction_smoke"
CUDA_VISIBLE_DEVICES=0 MAXF=1 NSEQ=1 OFFICIAL=1 \
  python3 reproducibility/run_davis_fullseq_portable.py
test -s "$OUT_DIR/results.json"
```

Record the command, output JSON SHA-256, package versions, and any model-download revision. Do not replace the full-sequence archive with this smoke output.

The portabilized driver differs from the canonical driver only by replacing hard-coded server paths with `MCVOS_ROOT`, `DAVIS_ROOT`, `SAM2_CKPT`, `SAM2_CFG`, and `OUT_DIR`. Its syntax was checked with `python3 -m py_compile`; the original canonical driver produced the archived 30-sequence result.

## What remains before public release

This runbook makes the primary evaluation reproducible in principle. Before submitting a public code package, run the preflight and one-sequence smoke workflow in a fresh environment, record the exact Hugging Face model artefact revision/hash, and preserve its output alongside the full-sequence archive.
