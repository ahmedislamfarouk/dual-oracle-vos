#!/usr/bin/env python3
"""UVOSAM segmentation stage without metaseg.

Faithful reimplementation of the repo's `eval_davis.py`: IDOL track boxes are
used as SAM box prompts, one mask per track per frame, thresholded at 0.5 and
composited into a DAVIS-style palette PNG with incremental instance ids.
metaseg is bypassed because installing it replaces the system PyTorch; the
`SahiAutoSegmentation().predict(..., model_type='vit_h', input_box=...,
multimask_output=False)` call it wraps is exactly SamPredictor with box prompts.
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

ap = argparse.ArgumentParser()
ap.add_argument('--davis_path', default='datasets/davis')
ap.add_argument('--track_res', required=True)
ap.add_argument('--save_path', required=True)
ap.add_argument('--sam_ckpt', default='ckpt/sam_vit_h_4b8939.pth')
ap.add_argument('--template', default='assets/template.png')
a = ap.parse_args()

template_palette = Image.open(a.template).getpalette()

sam = sam_model_registry['vit_h'](checkpoint=a.sam_ckpt).to('cuda').eval()
predictor = SamPredictor(sam)

tracks = json.load(open(a.track_res))
per_video = {}
for it in tracks:
    per_video.setdefault(it['video_id'], []).append(it['boxes'])

names = [l.strip() for l in open(f'{a.davis_path}/ImageSets/2017/val.txt') if l.strip()]
os.makedirs(a.save_path, exist_ok=True)

for vi, vid in enumerate(names):
    outdir = os.path.join(a.save_path, vid); os.makedirs(outdir, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(a.davis_path, 'JPEGImages/480p', vid, '*.*g')))
    trks = per_video.get(vid, [])
    for fi, ip in enumerate(imgs):
        img = np.array(Image.open(ip).convert('RGB'))
        h, w = img.shape[:2]
        boxes = []
        for tb in trks:
            b = tb[fi] if fi < len(tb) else None
            boxes.append([0, 0, 0, 0] if b is None else b)
        out = np.zeros((h, w), np.uint8)
        valid = [b for b in boxes if not (b[2] <= b[0] or b[3] <= b[1])]
        if valid:
            predictor.set_image(img)
            bt = torch.tensor(np.array(boxes, dtype=np.float32), device='cuda')
            bt = predictor.transform.apply_boxes_torch(bt, (h, w))
            with torch.no_grad():
                masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None,
                                                     boxes=bt, multimask_output=False)
            masks = masks[:, 0].cpu().numpy()
            real = 0
            for k, b in enumerate(boxes):
                if b[2] <= b[0] or b[3] <= b[1]:
                    continue
                m = (masks[k] > 0.5).astype(np.uint8)
                if m.sum() == 0:
                    continue
                real += 1
                out[m == 1] = real
        im = Image.fromarray(out, mode='P'); im.putpalette(template_palette)
        im.save(os.path.join(outdir, os.path.basename(ip).split('.')[0] + '.png'))
    print(f'[{vi+1}/{len(names)}] {vid}: {len(imgs)} frames, {len(trks)} tracks', flush=True)
print('DONE')
