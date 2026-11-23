"""Build a Kaggle GPU kernel folder that runs dovos jobs on two T4s (two shards each)."""
import base64, io, json, os, sys, tarfile
REL = os.path.expanduser("~/Desktop/dual-oracle-vos-release")

def code_b64(patch_fp16=False):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for sub in ("dovos", "scripts"):
            for root, _, files in os.walk(os.path.join(REL, sub)):
                if "__pycache__" in root: continue
                for f in files:
                    p = os.path.join(root, f); t.add(p, arcname=os.path.relpath(p, REL))
    return base64.b64encode(buf.getvalue()).decode()

RUNNER = r'''
import os, subprocess, sys, time, base64, io, tarfile, json
T0 = time.time(); DEADLINE = DEADLINE_H * 3600
def sh(c):
    print("$", c, flush=True); return subprocess.run(c, shell=True).returncode
os.environ["SAM2_BUILD_CUDA"] = "0"
sh("pip install -q --no-deps 'git+https://github.com/facebookresearch/sam2.git' 2>&1 | tail -2; pip install -q hydra-core iopath 'transformers==4.49.0' 'huggingface_hub<1.0' 2>&1 | tail -2")
CK = "/tmp/ckpt"; os.makedirs(CK, exist_ok=True)
sh(f"wget -q -O {CK}/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt")
CODE = "/tmp/code"; tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz").extractall(CODE)
if PATCH_FP16:
    import shutil; shutil.copytree(CODE, "/tmp/code16")
    p = "/tmp/code16/dovos/propagation.py"; s = open(p).read(); open(p, "w").write(s.replace("dtype=torch.bfloat16", "dtype=torch.float16"))
sh("git ls-remote https://github.com/facebookresearch/sam2.git HEAD; python -c \"import torch,transformers,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'tf',transformers.__version__,[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'bf16',torch.cuda.is_bf16_supported())\"")
for _p, _c in EXTRA_FILES.items(): open(_p, "w").write(_c)
OUT = "/kaggle/working/out"; LOG = "/kaggle/working/logs"; os.makedirs(OUT, exist_ok=True); os.makedirs(LOG, exist_ok=True)

def find(cond, top="/kaggle/input", depth=7):
    frontier = [top]
    for _ in range(depth):
        nxt = []
        for d in frontier:
            try: subs = [e.path for e in os.scandir(d) if e.is_dir()]
            except OSError: continue
            for s in subs:
                if cond(s): return s
                if os.path.basename(s) not in ("JPEGImages", "Annotations"): nxt.append(s)
        frontier = nxt
    return None
ROOTS = {}
dv = find(lambda s: os.path.basename(s) == "JPEGImages" and os.path.isdir(os.path.join(s, "480p")))
if dv: dv = os.path.join(dv, "480p")
if dv:
    base = os.path.dirname(os.path.dirname(dv)); r = "/tmp/davis"; os.makedirs(r, exist_ok=True)
    for k in ("JPEGImages", "Annotations"):
        if not os.path.exists(f"{r}/{k}"): os.symlink(f"{base}/{k}/480p", f"{r}/{k}")
    ROOTS["davis"] = r; ROOTS["davis_val"] = "/tmp/davis_val.txt"; open("/tmp/davis_val.txt", "w").write("\n".join('bike-packing blackswan bmx-trees breakdance camel car-roundabout car-shadow cows dance-twirl dog dogs-jump drift-chicane drift-straight goat gold-fish horsejump-high india judo kite-surf lab-coat libby loading mbike-trick motocross-jump paragliding-launch parkour pigs scooter-black shooting soapbox'.split()) + "\n")
lv = find(lambda s: os.path.basename(s) == "train" and os.path.isdir(os.path.join(s, "Annotations")) and os.path.isdir(os.path.join(s, "JPEGImages")) and "mose" not in s.lower())
if lv: ROOTS["lvos_train"] = lv
ms = find(lambda s: s.endswith(os.path.join("mose", "train")))
if ms: ROOTS["mose"] = ms; ROOTS["mose_dir"] = os.path.dirname(ms)
print("ROOTS", ROOTS, flush=True)
sh("ls /kaggle/input; ls /kaggle/input/* | head; ls /kaggle/input/*/* | head")

import torch
NG = max(1, torch.cuda.device_count()); print("GPUs", NG, flush=True)
procs_all = []
for name, script, args, code in JOBS:
    if time.time() - T0 > DEADLINE: print("deadline, skipping", name, flush=True); continue
    args = args.format(**ROOTS)
    procs = []
    for sh_i in range(NG):
        cmd = f"cd {code} && CUDA_VISIBLE_DEVICES={sh_i} python scripts/{script} {args} --checkpoint-dir {CK} --num-shards {NG} --shard {sh_i} --out {OUT}/{name}_s{sh_i}.json 2>&1 | tee {LOG}/{name}_s{sh_i}.log | sed -u 's/^/[s{sh_i}] /'"
        print("launch", cmd, flush=True); procs.append(subprocess.Popen(cmd, shell=True))
    t1 = time.time()
    while any(p.poll() is None for p in procs):
        if time.time() - T0 > DEADLINE:
            print("deadline hit, killing", name, flush=True)
            for p in procs: p.kill()
            break
        time.sleep(20)
    print(f"done {name} in {time.time()-t1:.0f}s rc={[p.returncode for p in procs]}", flush=True)
    sh(f"tail -n 3 {LOG}/{name}_s*.log")
print("all done", round(time.time() - T0), flush=True)
'''

def build(slug, jobs, sources, deadline_h=11.3, patch_fp16=False, kernel_sources=(), extra_files=None):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), slug); os.makedirs(d, exist_ok=True)
    src = f"CODE_B64 = {code_b64()!r}\nJOBS = {jobs!r}\nDEADLINE_H = {deadline_h}\nPATCH_FP16 = {patch_fp16}\nEXTRA_FILES = {(extra_files or {})!r}\n" + RUNNER
    open(os.path.join(d, f"{slug}.py"), "w").write(src)
    json.dump({"id": f"dynamicplays/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
               "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
               "enable_internet": True, "keywords": [], "dataset_sources": list(sources),
               "kernel_sources": list(kernel_sources), "competition_sources": [], "model_sources": [],
               "machine_shape": "NvidiaTeslaP100"}, open(os.path.join(d, "kernel-metadata.json"), "w"), indent=1)
    return d
