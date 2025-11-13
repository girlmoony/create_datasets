#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Excelの「class name」列を読み、source/<class>/ の PNG を
dest/train|test|val/<class>/ に 140/30/30 でコピー。
storeinfo.json は参照せず、ファイル名から (shop, lane) を抽出。
(店舗×レーン) バケットをラウンドロビンで巡回して多様性を最大化。

想定ファイル名例:
  [1.00]1304_011-02_2024-09-02_21-50-42-322_1.png
    -> shop=1304, lane=02  (011-02 の2番目)
  [0.94][0.99]1302_110_27-02-00_2025-11-04_20-47-18-482_1.png
    -> shop=1302, lane=02  (27-02-00 の2番目)
"""

import argparse
import logging
import random
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

DEFAULT_QUOTAS = {"train": 140, "test": 30, "val": 30}
SUBSETS = ["train", "test", "val"]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("copy_diverse_filename_only")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def read_excel(path: Path, sheet: str | None) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet or 0, engine="openpyxl")


def write_excel(path: Path, sheet: str | None, df: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as w:
        df.to_excel(w, index=False, sheet_name=sheet or "Sheet1")


def list_pngs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]


# --- ファイル名パーサ（store, lane） -----------------------------------------

SCORE_PREFIX_RE = re.compile(r"^(?:\[[^\]]+\])+")

def parse_shop_lane(filename: str) -> tuple[str | None, str | None]:
    """
    例:
      [1.00]1304_011-02_2024-... -> ('1304', '02')
      [0.94][0.99]1302_110_27-02-00_2025-... -> ('1302', '02')

    ルール:
      1) 先頭の [ ... ] タグを除去
      2) 先頭の '_' 区切りトークンが数字のみ -> shop
      3) 以降のトークンから、最初に出現する「数値-数値(-数値…)」の2番目要素を lane
    """
    base = Path(filename).name
    base = SCORE_PREFIX_RE.sub("", base)  # 先頭の [..] を除去
    base = base.split(".")[0]  # 拡張子除去

    parts = base.split("_")
    if not parts:
        return None, None

    # shop: 先頭が数字のみ
    shop = parts[0] if parts[0].isdigit() else None

    # lane 候補: 後続の最初の hyphen 数列の2番目
    lane = None
    for token in parts[1:]:
        if "-" in token:
            segs = token.split("-")
            # すべて数値で構成され、2番目が存在
            if len(segs) >= 2 and all(s.isdigit() for s in segs):
                lane = segs[1]
                break

    return shop, lane


# --- ラウンドロビン選定 & コピー ---------------------------------------------

def count_existing_and_mark_used(dest_root: Path, cls: str) -> tuple[dict[str, int], set[str]]:
    per_counts = {s: 0 for s in SUBSETS}
    used: set[str] = set()
    for s in SUBSETS:
        d = dest_root / s / cls
        if not d.exists():
            continue
        for p in list_pngs(d):
            per_counts[s] += 1
            used.add(p.name)
    return per_counts, used


def build_buckets(images: list[Path], logger: logging.Logger, rng: random.Random):
    """
    (shop, lane) をキーに画像をバケツ分け。
    パースできない場合は ("","") に入る（多様性には貢献しない）。
    """
    buckets = defaultdict(list)
    for img in images:
        shop, lane = parse_shop_lane(img.name)
        key = (shop or "", lane or "")
        buckets[key].append(img)

    # シャッフルして deque 化
    bucket_deques: dict[tuple[str, str], deque[Path]] = {}
    for k, lst in buckets.items():
        rng.shuffle(lst)
        bucket_deques[k] = deque(lst)
        logger.info(f"BUCKET {k}: {len(lst)} files")
    return bucket_deques


def select_images_diverse_rr(
    buckets: dict[tuple[str, str], deque[Path]],
    quotas_remaining: dict[str, int],
    used_names: set[str],
    rng: random.Random,
) -> dict[str, list[Path]]:
    """
    (shop, lane) バケットをラウンドロビンで巡回しつつ、split も巡回。
    - グローバル重複禁止（used_names）
    """
    selection = {s: [] for s in SUBSETS}
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    active_keys = [k for k in bucket_keys if buckets[k]]

    split_order = ["train", "test", "val"]
    split_idx = 0

    def has_quota():
        return any(quotas_remaining[s] > 0 for s in SUBSETS)

    while has_quota() and active_keys:
        # 次に埋める split
        tries = 0
        while quotas_remaining[split_order[split_idx]] <= 0 and tries < len(SUBSETS):
            split_idx = (split_idx + 1) % len(SUBSETS)
            tries += 1
        if quotas_remaining[split_order[split_idx]] <= 0:
            break
        target_split = split_order[split_idx]

        removed = []
        picked = False
        for k in active_keys:
            dq = buckets[k]
            while dq and dq[0].name in used_names:
                dq.popleft()
            if not dq:
                removed.append(k)
                continue
            cand = dq.popleft()
            if cand.name in used_names:
                continue
            selection[target_split].append(cand)
            used_names.add(cand.name)
            quotas_remaining[target_split] -= 1
            picked = True
            break

        if removed:
            active_keys = [k for k in active_keys if k not in removed]

        # 予備（貪欲）
        if not picked and active_keys:
            for k in list(active_keys):
                dq = buckets[k]
                while dq and dq[0].name in used_names:
                    dq.pop()
                if dq:
                    cand = dq.pop()
                    if cand.name in used_names:
                        continue
                    selection[target_split].append(cand)
                    used_names.add(cand.name)
                    quotas_remaining[target_split] -= 1
                    picked = True
                    break
                else:
                    active_keys.remove(k)

        split_idx = (split_idx + 1) % len(SUBSETS)
        if not active_keys and not picked:
            break

    return selection


def ensure_dirs(dest_root: Path, cls: str):
    for s in SUBSETS:
        (dest_root / s / cls).mkdir(parents=True, exist_ok=True)


def copy_with_skip(selection: dict[str, list[Path]], dest_root: Path, cls: str, logger: logging.Logger) -> dict[str, int]:
    counts = {s: 0 for s in SUBSETS}
    for s in SUBSETS:
        d = dest_root / s / cls
        d.mkdir(parents=True, exist_ok=True)
        for p in selection[s]:
            dst = d / p.name
            if dst.exists():
                logger.info(f"SKIP(存在): {dst}")
                continue
            shutil.copy2(p, dst)
            counts[s] += 1
            logger.info(f"COPY: {p} -> {dst}")
    return counts


def process_one_class(
    cls: str,
    source_root: Path,
    dest_root: Path,
    quotas: dict[str, int],
    seed: int,
    logger: logging.Logger,
) -> dict[str, int]:
    rng = random.Random(seed + hash(cls) % (10**6))

    src_dir = source_root / cls
    if not src_dir.exists():
        logger.warning(f"NOT FOUND: {src_dir}")
        return {s: 0 for s in SUBSETS}

    all_pngs = list_pngs(src_dir)
    if not all_pngs:
        logger.warning(f"NO PNGS: {src_dir}")
        return {s: 0 for s in SUBSETS}

    # 既存の割当と重複マーク
    existing_counts, used_names = count_existing_and_mark_used(dest_root, cls)
    quotas_remaining = {s: max(quotas[s] - existing_counts[s], 0) for s in SUBSETS}
    logger.info(
        f"[{cls}] 既存 train={existing_counts['train']}, test={existing_counts['test']}, val={existing_counts['val']} "
        f"/ 要求 train={quotas['train']}, test={quotas['test']}, val={quotas['val']} "
        f"/ 残 train={quotas_remaining['train']}, test={quotas_remaining['test']}, val={quotas_remaining['val']}"
    )

    # バケット構築 & 選定
    buckets = build_buckets(all_pngs, logger, rng)
    selection = select_images_diverse_rr(buckets, quotas_remaining, used_names, rng)

    # コピー
    ensure_dirs(dest_root, cls)
    copied_counts = copy_with_skip(selection, dest_root, cls, logger)

    final_counts = {s: existing_counts[s] + copied_counts[s] for s in SUBSETS}
    logger.info(f"[{cls}] 完了 train={final_counts['train']}, test={final_counts['test']}, val={final_counts['val']}")
    return final_counts


def main():
    ap = argparse.ArgumentParser(
        description="Excelのclass name列に基づき、source/<class>/ から PNG を (shop,lane) 多様性で選び、dest/train|test|val/<class>/ に 140/30/30 でコピーします。"
    )
    ap.add_argument("--excel", required=True, help="入力Excel（上書き保存）")
    ap.add_argument("--sheet", default=None, help="シート名（省略可）")
    ap.add_argument("--class-col", default="class name", help="クラス名列ヘッダ")
    ap.add_argument("--source", required=True, help="source ルート（直下に <class> フォルダ群）")
    ap.add_argument("--dest", required=True, help="出力ルート（train/test/val/<class> を作成）")
    ap.add_argument("--train", type=int, default=DEFAULT_QUOTAS["train"], help="train 枚数")
    ap.add_argument("--test", type=int, default=DEFAULT_QUOTAS["test"], help="test 枚数")
    ap.add_argument("--val", type=int, default=DEFAULT_QUOTAS["val"], help="val 枚数")
    ap.add_argument("--seed", type=int, default=42, help="乱数シード")
    ap.add_argument("--log", default="copy_diverse_filename_only.log", help="ログファイル")

    args = ap.parse_args()

    excel = Path(args.excel).resolve()
    source_root = Path(args.source).resolve()
    dest_root = Path(args.dest).resolve()

    logger = setup_logger(Path(args.log).resolve())
    dest_root.mkdir(parents=True, exist_ok=True)

    quotas = {"train": args.train, "test": args.test, "val": args.val}

    # Excel 読み
    try:
        df = read_excel(excel, args.sheet)
    except Exception as e:
        logger.error(f"Excel 読込失敗: {e}")
        raise SystemExit(1)

    if args.class_col not in df.columns:
        logger.error(f"列 '{args.class_col}' が見つかりません。列={list(df.columns)}")
        raise SystemExit(1)

    # 出力列（無ければ追加）
    for col in ["train_count", "test_count", "val_count", "total_selected"]:
        if col not in df.columns:
            df[col] = 0

    # 各クラス処理
    for idx, row in df.iterrows():
        cls = str(row[args.class_col]).strip()
        if not cls:
            continue
        counts = process_one_class(
            cls=cls,
            source_root=source_root,
            dest_root=dest_root,
            quotas=quotas,
            seed=args.seed,
            logger=logger,
        )
        df.at[idx, "train_count"] = counts["train"]
        df.at[idx, "test_count"] = counts["test"]
        df.at[idx, "val_count"] = counts["val"]
        df.at[idx, "total_selected"] = counts["train"] + counts["test"] + counts["val"]

    # Excel 書戻し
    try:
        write_excel(excel, args.sheet, df)
    except Exception as e:
        logger.error(f"Excel 書込失敗: {e}")
        raise SystemExit(1)

    logger.info("全クラス処理が完了しました。")


if __name__ == "__main__":
    main()

python copy_diverse_filename_only.py \
  --excel classes.xlsx \
  --sheet Sheet1 \
  --class-col "class name" \
  --source "D:/dataset/source" \
  --dest "D:/dataset/dest" \
  --train 140 --test 30 --val 30 \
  --seed 123 \
  --log copy.log
