#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Calibration Image Selector
-------------------------------
- OpenCV-backed gradients (Sobel/Laplacian) for speed (falls back to NumPy if OpenCV unavailable).
- Multiprocessing for parallel feature extraction (--workers).
- Candidate pool capping: --per-class-cap and --candidate-cap to bound runtime.
- Adjustable resize side (--max-side).
"""

import argparse
import csv
import os
import sys
import shutil
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image
from multiprocessing import Pool, cpu_count

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(root: Path, include_val: bool) -> List[Tuple[str, str, str]]:
    items = []
    for split in ["train", "val"]:
        if split == "val" and not include_val:
            continue
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            cls = cls_dir.name
            for fp in cls_dir.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                    items.append((split, cls, str(fp)))
    return items

def load_for_features(path: str, max_side: int) -> np.ndarray:
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read: {path}")
        h, w = img.shape[:2]
        if max_side and max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max_side and max(h, w) > max_side:
                scale = max_side / float(max(h, w))
                im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
            return arr

def color_hist_features(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 255), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)

def brightness_stats(arr: np.ndarray) -> np.ndarray:
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    mean = np.mean(gray)
    std = np.std(gray)
    return np.array([mean, std], dtype=np.float32)

def entropy_feature(arr: np.ndarray, bins: int = 32) -> np.ndarray:
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    hist, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    hist = np.clip(hist, 1e-12, None)
    ent = -np.sum(hist * np.log(hist))
    return np.array([ent], dtype=np.float32)

def edge_density_cv(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, 75.0)
    density = float((mag >= thr).mean())
    return np.array([density], dtype=np.float32)

def laplacian_var_cv(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=3)
    return np.array([float(np.var(lap))], dtype=np.float32)

def edge_density_np(arr: np.ndarray) -> np.ndarray:
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, 75.0)
    density = float((mag >= thr).mean())
    return np.array([density], dtype=np.float32)

def laplacian_var_np(arr: np.ndarray) -> np.ndarray:
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    lap = (
        -4 * gray +
        np.pad(gray, ((0,0),(1,0)), mode='edge')[:, :-1] +
        np.pad(gray, ((0,0),(0,1)), mode='edge')[:, 1:] +
        np.pad(gray, ((1,0),(0,0)), mode='edge')[:-1, :] +
        np.pad(gray, ((0,1),(0,0)), mode='edge')[1:, :]
    )
    return np.array([float(np.var(lap))], dtype=np.float32)

def extract_features_for_path(args):
    path, max_side = args
    arr = load_for_features(path, max_side=max_side)
    feats = [
        color_hist_features(arr, bins=16),
        brightness_stats(arr),
        entropy_feature(arr),
    ]
    if cv2 is not None:
        feats.append(edge_density_cv(arr))
        feats.append(laplacian_var_cv(arr))
    else:
        feats.append(edge_density_np(arr))
        feats.append(laplacian_var_np(arr))
    v = np.concatenate(feats, axis=0)
    return v

def normalize_features(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd

def class_medoid(features: np.ndarray) -> int:
    if features.shape[0] == 1:
        return 0
    mean = features.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(features - mean, axis=1)
    return int(np.argmin(dists))

def farthest_point_sampling(X: np.ndarray, k: int, initial_indices: Optional[List[int]] = None, seed: int = 0) -> List[int]:
    N = X.shape[0]
    rng = random.Random(seed)
    if k <= 0:
        return []
    if initial_indices is None or len(initial_indices) == 0:
        start = rng.randrange(N)
        selected = [start]
    else:
        selected = list(dict.fromkeys(initial_indices))
        if len(selected) == 0:
            selected = [rng.randrange(N)]
    dmin = np.full((N,), np.inf, dtype=np.float32)
    for s in selected:
        ds = np.linalg.norm(X - X[s], axis=1)
        dmin = np.minimum(dmin, ds)
    while len(selected) < min(k, N):
        idx = int(np.argmax(dmin))
        selected.append(idx)
        ds = np.linalg.norm(X - X[idx], axis=1)
        dmin = np.minimum(dmin, ds)
    return selected[:k]

@dataclass
class Item:
    split: str
    cls: str
    path: str
    feat: Optional[np.ndarray] = None

def cap_candidates(items: List[Item], per_class_cap: int, candidate_cap: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    cls_to_indices: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        cls_to_indices.setdefault(it.cls, []).append(i)
    picked: List[int] = []
    for cls, idxs in cls_to_indices.items():
        if per_class_cap > 0 and len(idxs) > per_class_cap:
            idxs = idxs[:]  # copy
            rng.shuffle(idxs)
            picked.extend(idxs[:per_class_cap])
        else:
            picked.extend(idxs)
    if candidate_cap > 0 and len(picked) > candidate_cap:
        rng.shuffle(picked)
        picked = picked[:candidate_cap]
    picked.sort()
    return picked

def select_calibration_set(items: List[Item], idx_pool: List[int], target: int, per_class_min: int, seed: int, max_side: int, workers: int) -> List[int]:
    paths = [items[i].path for i in idx_pool]
    args = [(p, max_side) for p in paths]
    workers = max(1, workers)
    workers = min(workers, len(args)) if len(args) > 0 else 1

    if workers > 1:
        with Pool(processes=workers) as pool:
            feats = pool.map(extract_features_for_path, args)
    else:
        feats = [extract_features_for_path(a) for a in args]

    for i, f in zip(idx_pool, feats):
        items[i].feat = f
    X = np.stack(feats, axis=0)
    Xn = normalize_features(X)

    cls_to_local: Dict[str, List[int]] = {}
    for j, i in enumerate(idx_pool):
        cls_to_local.setdefault(items[i].cls, []).append(j)
    initial_sel_local: List[int] = []
    for cls, locs in cls_to_local.items():
        take = min(per_class_min, len(locs)) if per_class_min > 0 else 0
        if take <= 0:
            continue
        feats_cls = Xn[locs, :]
        m_local = class_medoid(feats_cls)
        initial_sel_local.append(locs[m_local])

    k = min(target, len(idx_pool))
    sel_local = farthest_point_sampling(Xn, k=k, initial_indices=initial_sel_local, seed=seed)
    sel_global = [idx_pool[j] for j in sel_local]
    return sel_global

def write_manifest(items: List[Item], selected_idx: List[int], out_csv: Path):
    header = ["selected", "split", "class", "path"]
    selected_set = set(selected_idx)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, it in enumerate(items):
            sel = 1 if i in selected_set else 0
            w.writerow([sel, it.split, it.cls, it.path])

def maybe_copy_selected(items: List[Item], selected_idx: List[int], out_dir: Path, keep_tree: bool, symlink: bool, dryrun: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_set = set(selected_idx)
    for i, it in enumerate(items):
        if i not in selected_set:
            continue
        rel = Path(it.split) / it.cls / Path(it.path).name if keep_tree else Path(Path(it.path).name)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dryrun:
            print(f"[DRYRUN] -> {dst}")
            continue
        if symlink:
            try:
                if dst.exists():
                    dst.unlink()
                os.symlink(os.path.abspath(it.path), dst)
            except OSError:
                shutil.copy2(it.path, dst)
        else:
            shutil.copy2(it.path, dst)

def main():
    ap = argparse.ArgumentParser(description="Fast selection of representative calibration images")
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--include-val", action="store_true")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--per-class-min", type=int, default=1)
    ap.add_argument("--per-class-cap", type=int, default=0)
    ap.add_argument("--candidate-cap", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count()//2))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-manifest", type=str, default="calib_manifest.csv")
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--keep-tree", action="store_true")
    ap.add_argument("--symlink", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    items_raw = list_images(root, include_val=args.include_val)
    if not items_raw:
        print("[Error] no images found.", file=sys.stderr)
        sys.exit(1)
    items = [Item(split=s, cls=c, path=p) for (s, c, p) in items_raw]

    idx_pool = cap_candidates(items, per_class_cap=args.per_class_cap, candidate_cap=args.candidate_cap, seed=args.seed)
    print(f"Candidates: {len(idx_pool)} / total {len(items)} images "
          f"({'train+val' if args.include_val else 'train only'})")
    sel = select_calibration_set(items, idx_pool, target=args.target, per_class_min=args.per_class_min,
                                 seed=args.seed, max_side=args.max_side, workers=args.workers)

    out_csv = Path(args.out_manifest)
    write_manifest(items, sel, out_csv)
    print(f"[OK] manifest: {out_csv}  (selected={len(sel)})")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        maybe_copy_selected(items, sel, out_dir, keep_tree=args.keep_tree, symlink=args.symlink, dryrun=args.dryrun)
        print(f"[OK] images placed under: {out_dir}")

if __name__ == "__main__":
    main()


python calib_selector_fast.py \
  --root /path/to/dataset \
  --target 200 \
  --per-class-min 1 \
  --per-class-cap 60 \
  --candidate-cap 25000 \
  --max-side 256 \
  --workers 8 \
  --out-manifest calib_manifest.csv \
  --out-dir /path/to/calib_images --symlink --keep-tree

