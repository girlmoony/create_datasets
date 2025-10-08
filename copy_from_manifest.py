#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy selected images from calib_manifest.csv to a target folder.

CSV columns expected:
    selected,split,class,path

Usage:
  python copy_from_manifest.py \
      --manifest /path/to/calib_manifest.csv \
      --out-dir /path/to/selected_images \
      --keep-tree \
      --symlink \
      --dryrun
"""
import argparse
import csv
import os
import shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Copy selected images from calib_manifest.csv")
    ap.add_argument("--manifest", type=str, required=True, help="Path to calib_manifest.csv")
    ap.add_argument("--out-dir", type=str, required=True, help="Destination directory for selected images")
    ap.add_argument("--keep-tree", action="store_true", help="Preserve split/class folder structure")
    ap.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying")
    ap.add_argument("--dryrun", action="store_true", help="Print actions without writing files")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    with manifest.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"selected", "split", "class", "path"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            selected = str(row["selected"]).strip()
            if selected not in {"1", "true", "True"}:
                skipped += 1
                continue

            split = row["split"].strip()
            cls = row["class"].strip()
            src = Path(row["path"])

            if not src.exists():
                print(f"[WARN] missing file: {src}")
                continue

            rel = (Path(split) / cls / src.name) if args.keep_tree else Path(src.name)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            if args.dryrun:
                print(f"[DRYRUN] {src} -> {dst}")
            else:
                if args.symlink:
                    try:
                        if dst.exists():
                            dst.unlink()
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copy2(src, dst)
                else:
                    shutil.copy2(src, dst)
            copied += 1

    print(f"[DONE] copied={copied}, skipped(non-selected)={skipped}, dest={out_dir}")

if __name__ == "__main__":
    main()


# そのままコピー
python copy_from_manifest.py \
  --manifest /path/to/calib_manifest.csv \
  --out-dir /path/to/selected_images

# 元の構造を保持して配置（train/クラス名/…）
python copy_from_manifest.py \
  --manifest /path/to/calib_manifest.csv \
  --out-dir /path/to/selected_images \
  --keep-tree

# シンボリックリンクで配置（速い・省容量。権限が無ければ自動でコピーにフォールバック）
python copy_from_manifest.py \
  --manifest /path/to/calib_manifest.csv \
  --out-dir /path/to/selected_images \
  --keep-tree --symlink

# まずは動作確認だけ（ファイルは作らない）
python copy_from_manifest.py \
  --manifest /path/to/calib_manifest.csv \
  --out-dir /path/to/selected_images \
  --keep-tree --dryrun

