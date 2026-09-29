import sys, time, json, cv2
sys.path.insert(0, '/path/to/ow_vis/sam2')
sys.path.insert(0, '/path/to/ow_vis/src')
from pipeline import OWVISPipeline
from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
import torch

VIDEO = '/path/to/ow_vis/datasets/youtube_vis/sample_video.mp4'
SAM2_CKPT = '/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt'
DEVICE = 'cuda'
KNOWN_CATS = ['person', 'car', 'bicycle', 'dog', 'cat', 'bed', 'furniture', 'lamp']

cap = cv2.VideoCapture(VIDEO)
frames = []
while True:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()

bank = UnknownInstanceMemoryBank()
embedder = CropEmbedder().to(DEVICE)
pipeline = OWVISPipeline(
    sam2_checkpoint_path=SAM2_CKPT,
    sam2_config='sam2.1_hiera_base_plus',
    memory_bank=bank,
    crop_embedder=embedder,
    unknown_score_threshold=0.3,
    device=DEVICE
)

det_times, bank_times, total_times = [], [], []

for i, frame in enumerate(frames):
    bank.step_frame()
    
    t_total_start = time.perf_counter()
    
    # Time detection only
    t_det = time.perf_counter()
    result = pipeline.process_frame(frame, i, KNOWN_CATS)
    det_ms = (time.perf_counter() - t_det) * 1000
    
    total_ms = (time.perf_counter() - t_total_start) * 1000
    # bank time = total - detection (approx)
    bank_ms = max(0.0, total_ms - det_ms)
    
    det_times.append(det_ms)
    bank_times.append(bank_ms)
    total_times.append(total_ms)

avg_det = sum(det_times)/len(det_times)
avg_bank = sum(bank_times)/len(bank_times)
avg_total = sum(total_times)/len(total_times)
fps = 1000.0 / avg_total

timing_result = {
    'n_frames': len(frames),
    'avg_detection_ms': round(avg_det, 2),
    'avg_memory_bank_ms': round(avg_bank, 2),
    'avg_total_ms': round(avg_total, 2),
    'fps': round(fps, 2),
    'pct_detection': round(100*avg_det/avg_total, 1),
    'pct_memory_bank': round(100*avg_bank/avg_total, 1)
}
print(json.dumps(timing_result, indent=2))
with open('/path/to/ow_vis/results/timing.json', 'w') as f:
    json.dump(timing_result, f, indent=2)
print('Saved timing.json')
