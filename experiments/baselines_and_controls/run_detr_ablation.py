import sys, json, time, os
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
from pipeline import OWVISPipeline
from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
from metrics import compute_id_consistency
import torch, cv2

VIDEO = "/path/to/ow_vis/datasets/youtube_vis/sample_video.mp4"
CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cuda"

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
print("Loaded", len(frames), "frames")

def run_cond(label, no_bank=False, no_ema=False, no_promo=False):
    bank = UnknownInstanceMemoryBank(promote_after_obs=999999 if no_promo else 8)
    if no_ema:
        bank.ema_momentum = 0.0
    emb = CropEmbedder().to(DEVICE)
    try:
        ck = torch.load("/path/to/ow_vis/checkpoints/crop_embedder_best.pt", map_location=DEVICE)
        emb.load_state_dict(ck["model_state_dict"])
    except:
        pass
    pipe = OWVISPipeline(CKPT, "sam2.1_hiera_base_plus", bank, emb,
                         unknown_score_threshold=0.5, use_owdetr=True, device=DEVICE)
    all_ids = []
    tot_known = 0
    tot_unknown = 0
    det_times = []
    for i, frame in enumerate(frames):
        bank.step_frame()
        t0 = time.perf_counter()
        res = pipe.process_frame(frame, i, [])
        dt = (time.perf_counter() - t0) * 1000
        det_times.append(dt)
        ids = res.get("unknown_ids", [])
        if no_bank:
            ids = list(range(bank._next_id, bank._next_id + len(ids)))
            bank._next_id += len(ids)
        all_ids.append(ids)
        tot_unknown += len(ids)
        tot_known += len(res.get("known_boxes", []))
    unique = len(set(x for ids in all_ids for x in ids))
    consist = compute_id_consistency(all_ids)
    avg_ms = round(sum(det_times) / len(det_times), 1)
    r = {"condition": label, "total_known": tot_known, "total_unknown": tot_unknown,
         "unique_unknown_ids": unique, "id_consistency": round(consist, 4), "avg_det_ms": avg_ms}
    print(label, "known=" + str(tot_known), "unk=" + str(tot_unknown),
          "unique=" + str(unique), "consist=" + str(round(consist, 4)), str(avg_ms) + "ms")
    return r

print("\n=== DETR Open-World Ablation (bedroom, no text queries) ===")
results = []
results.append(run_cond("no_memory_bank", no_bank=True))
results.append(run_cond("bank_no_ema", no_ema=True))
results.append(run_cond("bank_no_promotion", no_promo=True))
results.append(run_cond("full_system"))

os.makedirs("/path/to/ow_vis/results/detr_ablation", exist_ok=True)
with open("/path/to/ow_vis/results/detr_ablation/ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved ablation.json")
