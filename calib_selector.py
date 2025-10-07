#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration Image Selector
--------------------------
Select a representative subset of images (e.g., 100–200) for post-training quantization calibration.

Features:
- Scans a dataset organized as:
    <root>/train/<class_name>/*.jpg|png|...
    <root>/val/<class_name>/*.jpg|png|...
- Optional inclusion of val split (images are unlabeled for calibration; labels are used only for per-class coverage).
- Extracts lightweight visual features (no heavy dependencies): 
    * Color histograms (per-channel 16 bins) [48 dims]
    * Brightness mean & std (2 dims)
    * Entropy (1 dim)
    * Edge density (via simple Sobel) (1 dim)
    * Blur (variance of Laplacian) (1 dim)
  -> Total ~53D feature vector per image.
- Ensures per-class minimum selection (default 1).
- Greedy farthest-point sampling (k-Center) over normalized features to maximize diversity.
- Optionally copies selected images to a target directory or writes a CSV manifest with metrics.

Requirements: Python 3.8+, Pillow, numpy
(If OpenCV is available, you *can* modify to use it, but this script avoids that dependency.)

Usage:
    python calib_selector.py \
        --root /path/to/dataset \
        --include-val \
        --target 200 \
        --per-class-min 1 \
        --out-manifest calib_manifest.csv \
        --out-dir /path/to/calib_images \
        --copy

Notes:
- For reproducibility, you can set --seed 42
- Supported image extensions: .jpg .jpeg .png .bmp .tiff .webp (configurable)
"""

import argparse
import csv
import os
import sys
import shutil
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

# -----------------------------
# Utilities
# -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(root: Path, include_val: bool) -> List[Tuple[str, str, str]]:
    """
    Returns list of (split, class_name, filepath)
    Expects directories: root/train/<class>/*, and optionally root/val/<class>/*
    """
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

def pil_load_rgb(path: str, max_side: int = 384) -> Image.Image:
    """
    Load image with Pillow, convert to RGB.
    Optional: resize so that max(H, W) <= max_side to keep feature extraction fast/consistent.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h)) if max_side else 1.0
        if scale < 1.0:
            im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        return im

def np_from_pil(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32)  # HxWx3 in [0..255]

# -----------------------------
# Feature extraction
# -----------------------------

def color_hist_features(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 255), density=True)
        feats.append(hist.astype(np.float32))
    return np.concatenate(feats, axis=0)  # 48D

def brightness_stats(arr: np.ndarray) -> np.ndarray:
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mean = np.mean(gray)
    std = np.std(gray)
    return np.array([mean, std], dtype=np.float32)

def entropy_feature(arr: np.ndarray, bins: int = 32) -> np.ndarray:
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hist, _ = np.histogram(gray, bins=bins, range=(0, 255), density=True)
    hist = np.clip(hist, 1e-12, None)
    ent = -np.sum(hist * np.log(hist))
    return np.array([ent], dtype=np.float32)

def sobel_edge_density(arr: np.ndarray) -> np.ndarray:
    # Simple Sobel filters
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = conv2d_same(gray, Kx)
    gy = conv2d_same(gray, Ky)
    mag = np.sqrt(gx * gx + gy * gy)
    # Edge density: ratio of strong edges
    thr = np.percentile(mag, 75.0)
    density = float((mag >= thr).mean())
    return np.array([density], dtype=np.float32)

def laplacian_var(arr: np.ndarray) -> np.ndarray:
    # Blur measure: variance of Laplacian
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    K = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    lap = conv2d_same(gray, K)
    return np.array([float(np.var(lap))], dtype=np.float32)

def conv2d_same(im: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """naive 2D conv with reflect padding, same size"""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(im, dtype=np.float32)
    # flip kernel for convolution
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    H, W = im.shape
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = float(np.sum(region * k))
    return out

def extract_features(img: Image.Image) -> np.ndarray:
    arr = np_from_pil(img)
    feats = [
        color_hist_features(arr, bins=16),
        brightness_stats(arr),
        entropy_feature(arr),
        sobel_edge_density(arr),
        laplacian_var(arr),
    ]
    v = np.concatenate(feats, axis=0)
    return v  # ~53D

# -----------------------------
# Selection
# -----------------------------

def normalize_features(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd

def class_medoid(features: np.ndarray) -> int:
    """
    Return index of medoid (closest to class mean in L2).
    """
    if features.shape[0] == 1:
        return 0
    mean = features.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(features - mean, axis=1)
    return int(np.argmin(dists))

def farthest_point_sampling(X: np.ndarray, k: int, initial_indices: Optional[List[int]] = None, seed: int = 0) -> List[int]:
    """
    Greedy k-center selection in Euclidean space.
    X: (N, D) normalized features
    k: number to select
    initial_indices: preseeded indices (e.g., per-class medoids)
    """
    N = X.shape[0]
    rng = random.Random(seed)
    if k <= 0:
        return []
    if initial_indices is None or len(initial_indices) == 0:
        start = rng.randrange(N)
        selected = [start]
    else:
        selected = list(dict.fromkeys(initial_indices))  # dedup, keep order
        if len(selected) == 0:
            selected = [rng.randrange(N)]
    # Initialize min distances
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

# -----------------------------
# Orchestration
# -----------------------------

@dataclass
class Item:
    split: str
    cls: str
    path: str
    feat: Optional[np.ndarray] = None

def select_calibration_set(
    items: List[Item],
    target: int = 200,
    per_class_min: int = 1,
    seed: int = 42
) -> List[int]:
    """
    Returns indices of 'items' to select.
    Strategy:
        1) Compute features for all images.
        2) Ensure per-class_min by taking each class medoid (closest to class mean) up to per_class_min.
        3) Apply greedy farthest-point sampling globally until target.
    """
    # 1) features
    feats = []
    for it in items:
        img = pil_load_rgb(it.path, max_side=384)
        it.feat = extract_features(img)
        feats.append(it.feat)
    X = np.stack(feats, axis=0)
    Xn = normalize_features(X)

    # 2) per-class medoids
    cls_to_indices: Dict[str, List[int]] = {}
    for idx, it in enumerate(items):
        cls_to_indices.setdefault(it.cls, []).append(idx)

    initial_sel: List[int] = []
    rng = random.Random(seed)
    for cls, idxs in cls_to_indices.items():
        if per_class_min <= 0:
            continue
        # If class has fewer than per_class_min, take all
        take = min(per_class_min, len(idxs))
        feats_cls = np.stack([items[i].feat for i in idxs], axis=0)
        # First medoid
        m = class_medoid(feats_cls)
        initial_sel.append(idxs[m])
        # If need more than 1 per class, add additional diverse picks within class
        if take > 1:
            # simple FPS within the class for remaining
            Xc = normalize_features(feats_cls)
            sel_local = farthest_point_sampling(Xc, k=take, initial_indices=[m], seed=seed)
            for s in sel_local[1:]:
                initial_sel.append(idxs[s])

    # 3) global diversity via FPS
    k = min(target, len(items))
    sel = farthest_point_sampling(Xn, k=k, initial_indices=initial_sel, seed=seed)
    return sel

def write_manifest(items: List[Item], selected_idx: List[int], out_csv: Path):
    header = [
        "selected", "split", "class", "path"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        selected_set = set(selected_idx)
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
    ap = argparse.ArgumentParser(description="Select representative calibration images")
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing train/ and optionally val/")
    ap.add_argument("--include-val", action="store_true", help="Include val split images for selection")
    ap.add_argument("--target", type=int, default=200, help="Target number of calibration images to select")
    ap.add_argument("--per-class-min", type=int, default=1, help="Minimum images per class to include (ensure coverage)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for tie-breaking and initial picks")
    ap.add_argument("--out-manifest", type=str, default="calib_manifest.csv", help="CSV path to write selection manifest")
    ap.add_argument("--out-dir", type=str, default="", help="If set, copy/symlink selected images to this directory")
    ap.add_argument("--keep-tree", action="store_true", help="Preserve split/class folder structure under out-dir")
    ap.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying (falls back to copy if unsupported)")
    ap.add_argument("--dryrun", action="store_true", help="Only print planned copies")
    ap.add_argument("--exts", type=str, default=",".join(sorted(IMG_EXTS)), help="Comma-separated allowed extensions")
    args = ap.parse_args()

    global IMG_EXTS
    IMG_EXTS = {e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
                for e in args.exts.split(",") if e.strip()}

    root = Path(args.root)
    if not root.exists():
        print(f"[Error] root not found: {root}", file=sys.stderr)
        sys.exit(1)

    items_raw = list_images(root, include_val=args.include_val)
    if not items_raw:
        print("[Error] no images found. Check --root and directory structure.", file=sys.stderr)
        sys.exit(1)

    items: List[Item] = [Item(split=s, cls=c, path=p) for (s, c, p) in items_raw]

    print(f"Found {len(items)} images in {len(set([c for _, c, _ in items_raw]))} classes "
          f"({'train+val' if args.include_val else 'train only'}).")

    sel_idx = select_calibration_set(
        items,
        target=args.target,
        per_class_min=args.per_class_min,
        seed=args.seed
    )

    out_csv = Path(args.out_manifest)
    write_manifest(items, sel_idx, out_csv)
    print(f"[OK] Wrote manifest: {out_csv} (selected {len(sel_idx)} images)")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        maybe_copy_selected(items, sel_idx, out_dir, keep_tree=args.keep_tree, symlink=args.symlink, dryrun=args.dryrun)
        print(f"[OK] Placed selected images under: {out_dir}")

if __name__ == "__main__":
    main()
