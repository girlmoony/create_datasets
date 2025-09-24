# -*- coding: utf-8 -*-
"""
全サブフォルダ横断の画像重複チェック（内容ベース）
- ファイル名やフォルダ名は無視、バイト内容で判定
- 段階的比較: size → md5（→ 任意で sha256）
- 複数ルートを指定可能（例：286 と 330 を同時に検査）
- 出力: duplicate_groups.csv / duplicate_files.csv / unique_files_sample.csv（任意）

標準ライブラリのみ（追加インストール不要）
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

# ---------- ユーティリティ ----------

def iter_image_files(roots: List[Path], exts: set) -> List[Path]:
    files = []
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                p = Path(dirpath) / name
                if not p.is_file():
                    continue
                if exts and p.suffix.lower() not in exts:
                    continue
                files.append(p)
    return files

def md5sum(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def sha256sum(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def human_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024 or unit == 'TB':
            return f"{n:.1f}{unit}"
        n /= 1024.0

# ---------- 本体 ----------

def main():
    ap = argparse.ArgumentParser(description="内容ベースの画像重複検出（全サブフォルダ横断）")
    ap.add_argument("--roots", nargs="+", required=True, help="検査するルートディレクトリ（複数可）")
    ap.add_argument("--out-dir", required=True, help="CSV 出力先ディレクトリ")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.gif,.tif,.tiff,.webp",
                    help="対象拡張子（小文字・カンマ区切り）例: .jpg,.png,.webp")
    ap.add_argument("--deep-sha256", action="store_true",
                    help="MD5一致グループを SHA256 でも確認（より厳密だが遅くなる）")
    ap.add_argument("--unique-sample", type=int, default=0,
                    help="一意ファイルのサンプル件数をCSV出力（0で出力しない）")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    # 1) 収集
    all_files = iter_image_files(roots, exts)
    if not all_files:
        print("[INFO] 画像が見つかりませんでした。拡張子やパスを確認してください。")
        return

    # 2) サイズでプレグループ化（高速フィルタ）
    by_size: Dict[int, List[Path]] = defaultdict(list)
    for p in all_files:
        try:
            sz = p.stat().st_size
        except FileNotFoundError:
            continue
        by_size[sz].append(p)

    # 3) サイズが同じグループだけ MD5 計算
    md5_groups: Dict[Tuple[int, str], List[Path]] = defaultdict(list)
    for sz, files in by_size.items():
        if len(files) == 1:
            # サイズ一意 → 重複の可能性低いのでスキップ（必要なら md5 取ってもOK）
            md5_groups[(sz, f"__unique__:{files[0].name}")].append(files[0])
            continue
        for p in files:
            try:
                h = md5sum(p)
            except Exception:
                # 読めない等は別扱いにしておく
                h = "__read_error__"
            md5_groups[(sz, h)].append(p)

    # 4) （任意）MD5一致グループを SHA256 で再分割
    final_groups: Dict[str, List[Path]] = {}
    group_id_counter = 1

    for (sz, md5h), files in md5_groups.items():
        if md5h.startswith("__") or len(files) == 1 or not args.deep_sha256:
            gid = f"G{group_id_counter:06d}"
            final_groups[gid] = files
            group_id_counter += 1
        else:
            # sha256 で細分化
            by_sha: Dict[str, List[Path]] = defaultdict(list)
            for p in files:
                try:
                    s = sha256sum(p)
                except Exception:
                    s = "__read_error__"
                by_sha[s].append(p)
            for s, fs in by_sha.items():
                gid = f"G{group_id_counter:06d}"
                final_groups[gid] = fs
                group_id_counter += 1

    # 5) 出力用のグループ統計と重複抽出
    dup_groups = []      # 複数ファイルを含むグループのみ
    dup_files_rows = []  # 重複グループに属する全ファイル

    unique_sample_rows = []
    unique_pool = []

    for gid, files in final_groups.items():
        # サイズは同じはずだが安全に再取得
        try:
            size = files[0].stat().st_size
        except Exception:
            size = -1

        if len(files) == 1:
            # 一意グループ
            unique_pool.append(files[0])
            continue

        # 重複グループ
        wasted = size * (len(files) - 1)
        dup_groups.append({
            "group_id": gid,
            "count": len(files),
            "file_size_bytes": size,
            "file_size_human": human_bytes(size),
            "potential_saving_bytes": wasted,
            "potential_saving_human": human_bytes(wasted),
        })
        for p in files:
            dup_files_rows.append({
                "group_id": gid,
                "path": str(p),
                "file_name": p.name,
                "size_bytes": size,
            })

    # 6) CSV 出力
    groups_csv = out_dir / "duplicate_groups.csv"
    files_csv  = out_dir / "duplicate_files.csv"
    with groups_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "group_id", "count",
            "file_size_bytes", "file_size_human",
            "potential_saving_bytes", "potential_saving_human",
        ])
        w.writeheader()
        for row in sorted(dup_groups, key=lambda r: (-r["potential_saving_bytes"], r["group_id"])):
            w.writerow(row)

    with files_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "path", "file_name", "size_bytes"])
        w.writeheader()
        for row in sorted(dup_files_rows, key=lambda r: (r["group_id"], r["path"])):
            w.writerow(row)

    if args.unique_sample and unique_pool:
        sample_csv = out_dir / "unique_files_sample.csv"
        with sample_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["path", "file_name", "size_bytes"])
            w.writeheader()
            for p in unique_pool[:args.unique_sample]:
                try:
                    sz = p.stat().st_size
                except Exception:
                    sz = -1
                w.writerow({"path": str(p), "file_name": p.name, "size_bytes": sz})

    # 7) サマリ表示
    total = len(all_files)
    dup_groups_n = len(dup_groups)
    dup_files_n = sum(g["count"] for g in dup_groups)
    saving = sum(g["potential_saving_bytes"] for g in dup_groups)

    print("========== SUMMARY ==========")
    print(f"Scanned files : {total}")
    print(f"Duplicate grps: {dup_groups_n}")
    print(f"Files in dups : {dup_files_n} (of {total})")
    print(f"Potential save: {human_bytes(saving)}")
    print(f"CSV written   : {groups_csv}")
    print(f"                {files_csv}")
    if args.unique_sample and unique_pool:
        print(f"                {out_dir / 'unique_files_sample.csv'}")

if __name__ == "__main__":
    main()


python find_image_duplicates.py ^
  --roots "D:\datasets\286" "D:\datasets\330" ^
  --out-dir "D:\dup_report" ^
  --exts .jpg,.jpeg,.png,.bmp,.gif,.tif,.tiff,.webp

# SHA256での二重確認もしたい場合（厳密だが少し遅い）
python find_image_duplicates.py --roots "D:\datasets\286" "D:\datasets\330" --out-dir ".\report" --deep-sha256
