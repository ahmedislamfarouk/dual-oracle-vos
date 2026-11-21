"""Pre-registered runs R1 (fresh capped MOSE), R3 (Mask R-CNN) and a DAVIS hardware check on Modal A10G GPUs.

    modal run --detach dovos_modal.py            # prepare data, run every job, write /out/*.json to the volume
    modal volume get dovos-runs out ./out        # fetch results
"""
import json
import os
import subprocess
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.expanduser("~/Desktop/dual-oracle-vos-release")
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
GPU = "A10G"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "unzip", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.7.1", "torchvision==0.22.1")
    .pip_install("transformers==4.49.0", "huggingface_hub<1.0", "hydra-core", "iopath", "opencv-python-headless",
                 "pillow", "numpy<2", "scipy", "stream-unzip", "tqdm")
    .run_commands(f"SAM2_BUILD_CUDA=0 pip install --no-deps git+https://github.com/facebookresearch/sam2.git@{SAM2_COMMIT}",
                  "mkdir -p /ckpt && curl -sSL -o /ckpt/sam2.1_hiera_large.pt "
                  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
                  "python -c \"from transformers import DetrForObjectDetection as D, DetrImageProcessor as P; "
                  "P.from_pretrained('facebook/detr-resnet-50', revision='no_timm'); "
                  "D.from_pretrained('facebook/detr-resnet-50', revision='no_timm')\"",
                  "python -c \"from torchvision.models.detection import maskrcnn_resnet50_fpn as m, "
                  "MaskRCNN_ResNet50_FPN_Weights as W; m(weights=W.DEFAULT)\"")
    .add_local_dir(os.path.join(REL, "dovos"), "/code/dovos", ignore=["__pycache__"])
    .add_local_dir(os.path.join(REL, "scripts"), "/code/scripts", ignore=["__pycache__"])
    .add_local_file(os.path.join(HERE, "mose_lists.json"), "/code/mose_lists.json")
)
vol = modal.Volume.from_name("dovos-runs", create_if_missing=True)
app = modal.App("dovos-prereg", image=image)
V = "/vol"

DAVIS_VAL = ("bike-packing blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog dogs-jump "
             "drift-chicane drift-straight goat gold-fish horsejump-high india judo kite-surf lab-coat libby loading "
             "mbike-trick motocross-jump paragliding-launch parkour pigs scooter-black shooting soapbox").split()

MF = f"--root {V}/mose/train --video-list {V}/mose/fresh240.txt --max-frames 60"
MP = f"--root {V}/mose/train --video-list {V}/mose/paper120.txt --max-frames 60 --detector maskrcnn"
DV = f"--root {V}/davis --video-list {V}/davis/val.txt --tiou-stride 4"
JOBS = [  # name, script, args, shards
    ("mose_qrd", "run_qrd.py", MF + " --tiou-stride 4", 10),
    ("mose_o1", "run_seeding.py", MF + " --seeding o1", 4),
    ("mose_o4b1", "run_seeding.py", MF + " --seeding o4 --seeds 1 --frames best", 6),
    ("mose_o4b3", "run_seeding.py", MF + " --seeding o4 --seeds 3 --frames best", 6),
    ("davis_qrd", "run_qrd.py", DV, 5),
    ("mrcnn_qrd", "run_qrd.py", MP + " --tiou-stride 4", 6),
    ("mrcnn_o4b1", "run_seeding.py", MP + " --seeding o4 --seeds 1 --frames best", 4),
]


def sh(c):
    print("$", c[:400], flush=True)
    r = subprocess.run(c, shell=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {c[:200]}")


@app.function(volumes={V: vol}, timeout=3 * 3600, cpu=8, memory=16384)
def prep_mose():
    if os.path.exists(f"{V}/mose/READY"):
        return "cached"
    lists = json.load(open("/code/mose_lists.json"))
    keep = sorted({v for l in lists.values() for v in l})
    pats = " ".join(f"'*/JPEGImages/{v}/000[0-5][0-9].jpg' '*/Annotations/{v}/000[0-5][0-9].png'" for v in keep)
    open("/tmp/su.py", "w").write(
        "import sys\nfrom stream_unzip import stream_unzip\n"
        "def ch():\n    while True:\n        b=sys.stdin.buffer.read(1<<22)\n        if not b: return\n        yield b\n"
        "for n,s,cs in stream_unzip(ch()):\n    if n==b'MOSE_release/train.tar.gz':\n"
        "        for c in cs: sys.stdout.buffer.write(c)\n    else:\n        for c in cs: pass\n")
    os.makedirs("/tmp/mx", exist_ok=True)
    t0 = time.time()
    sh("curl -sSL --retry 5 https://huggingface.co/datasets/FudanCVL/MOSE/resolve/main/MOSE_release.zip"
       f" | python /tmp/su.py | tar -xz -C /tmp/mx --wildcards {pats}")
    src = next(os.path.dirname(r) for r, _, _ in os.walk("/tmp/mx") if os.path.basename(r) == "JPEGImages")
    os.makedirs(f"{V}/mose", exist_ok=True)
    sh(f"rm -rf {V}/mose/train && cp -r {src} {V}/mose/train")
    for k, l in lists.items():
        open(f"{V}/mose/{k}.txt", "w").write("\n".join(l) + "\n")
    n = {v: len(os.listdir(f"{V}/mose/train/JPEGImages/{v}")) for v in keep if os.path.isdir(f"{V}/mose/train/JPEGImages/{v}")}
    open(f"{V}/mose/READY", "w").write(json.dumps({"videos": len(n), "frames": sum(n.values()), "sec": time.time() - t0}))
    vol.commit()
    return f"mose: {len(n)} videos, {sum(n.values())} frames, {time.time() - t0:.0f}s"


@app.function(volumes={V: vol}, timeout=2 * 3600, cpu=4)
def prep_davis():
    if os.path.exists(f"{V}/davis/READY"):
        return "cached"
    sh("curl -sSL --retry 5 -o /tmp/d.zip https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip")
    sh("cd /tmp && unzip -q d.zip")
    os.makedirs(f"{V}/davis", exist_ok=True)
    for k in ("JPEGImages", "Annotations"):
        sh(f"rm -rf {V}/davis/{k} && mkdir -p {V}/davis/{k} && cp -r /tmp/DAVIS/{k}/480p/. {V}/davis/{k}/")
    open(f"{V}/davis/val.txt", "w").write("\n".join(DAVIS_VAL) + "\n")
    open(f"{V}/davis/READY", "w").write("ok")
    vol.commit()
    return "davis ok"


@app.function(volumes={V: vol}, gpu=GPU, timeout=12 * 3600, cpu=4, memory=32768, retries=1)
def run_shard(name: str, script: str, args: str, shard: int, nshards: int):
    vol.reload()
    out = f"{V}/out/{name}_s{shard}.json"
    if os.path.exists(out) and json.load(open(out)).get("complete"):
        return f"{name} s{shard}: cached"
    os.makedirs(f"{V}/out", exist_ok=True)
    tmp = f"/tmp/{name}_s{shard}.json"
    t0 = time.time()
    p = subprocess.Popen(f"cd /code && python scripts/{script} {args} --checkpoint-dir /ckpt --num-shards {nshards} "
                         f"--shard {shard} --out {tmp}", shell=True)
    last = time.time()
    while p.poll() is None:
        time.sleep(30)
        if time.time() - last > 900 and os.path.exists(tmp):   # checkpoint partial results every 15 min
            sh(f"cp {tmp} {out}"); vol.commit(); last = time.time()
    if os.path.exists(tmp):
        sh(f"cp {tmp} {out}")
    vol.commit()
    return f"{name} s{shard}/{nshards}: rc={p.returncode} {time.time() - t0:.0f}s"


@app.function(volumes={V: vol}, timeout=24 * 3600)
def orchestrate(only: str = ""):
    print(prep_mose.remote(), flush=True)
    print(prep_davis.remote(), flush=True)
    calls = [run_shard.spawn(n, s, a, i, k) for n, s, a, k in JOBS if not only or n in only.split(",") for i in range(k)]
    for c in calls:
        try:
            print(c.get(), flush=True)
        except Exception as e:
            print("shard failed:", repr(e)[:300], flush=True)
    return "done"


@app.local_entrypoint()
def main(only: str = ""):
    print(orchestrate.remote(only))


@app.function(volumes={V: vol}, timeout=3600)
def smoke():
    print(prep_davis.remote(), flush=True)
    return run_shard.remote("smoke_davis", "run_qrd.py", DV + " --max-videos 2", 0, 1)
