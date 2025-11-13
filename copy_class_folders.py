#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import shutil
from pathlib import Path
import pandas as pd

SUBSETS = ["train", "val", "test"]

def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("copy_classes")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def iter_immediate_pngs(folder: Path):
    """フォルダ直下の .png（大文字拡張子も含む）のみを列挙"""
    if not folder.exists() or not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]

def copy_subset_folder(src_subset_dir: Path, dst_subset_dir: Path, logger: logging.Logger):
    """
    src_subset_dir: 例) /source/train/CLASS_A
    dst_subset_dir: 例) /dest/train/CLASS_A
    同名フォルダが既に存在する場合はスキップ（ログ出力）。
    存在しない場合は作成し、直下のPNGのみコピー。
    戻り値: dst側に存在するPNG枚数（スキップ時も数える）
    """
    if dst_subset_dir.exists():
        logger.info(f"SKIP: 既に存在するためスキップ: {dst_subset_dir}")
    else:
        # コピー先フォルダを作成
        dst_subset_dir.mkdir(parents=True, exist_ok=True)
        # 直下PNGのみコピー
        pngs = iter_immediate_pngs(src_subset_dir)
        for f in pngs:
            shutil.copy2(f, dst_subset_dir / f.name)
        logger.info(f"COPIED: {src_subset_dir} -> {dst_subset_dir} ({len(pngs)} files)")
    # いずれにせよ、dest 側の直下PNG数を返す
    return len(iter_immediate_pngs(dst_subset_dir)) if dst_subset_dir.exists() else 0

def process_row(class_name: str, source_root: Path, dest_root: Path, logger: logging.Logger):
    """
    1クラス名について train/val/test を処理し、各枚数を返す。
    例: {'train': 10, 'val': 2, 'test': 0}
    """
    counts = {s: 0 for s in SUBSETS}
    if not class_name or str(class_name).strip() == "":
        return counts  # 空欄はスキップ

    cls = str(class_name).strip()
    for subset in SUBSETS:
        src_dir = source_root / subset / cls
        dst_dir = dest_root / subset / cls

        if not src_dir.exists() or not src_dir.is_dir():
            logger.info(f"NOT FOUND: {src_dir} が存在しません。")
            counts[subset] = 0
            continue

        counts[subset] = copy_subset_folder(src_dir, dst_dir, logger)

    return counts

def main():
    parser = argparse.ArgumentParser(
        description="Excelのclass name列を基に、source直下のtrain/val/testの同名フォルダをdestへコピーし、結果枚数をExcelに出力します。"
    )
    parser.add_argument("--excel", required=True, help="入力Excelファイルパス（上書き保存されます）")
    parser.add_argument("--sheet", default=None, help="シート名（未指定なら先頭シート）")
    parser.add_argument("--class-col", default="class name", help="クラス名列の見出し（デフォルト: 'class name'）")
    parser.add_argument("--result-col", default="result", help="結果を書き込む列名（デフォルト: 'result'）")
    parser.add_argument("--source", required=True, help="source root（train/val/testが直下にあるディレクトリ）")
    parser.add_argument("--dest", required=True, help="dest root（train/val/testを作成します）")
    parser.add_argument("--log", default="copy_classes.log", help="ログ出力先ファイル（デフォルト: copy_classes.log）")

    args = parser.parse_args()

    source_root = Path(args.source).resolve()
    dest_root = Path(args.dest).resolve()
    log_path = Path(args.log).resolve()

    logger = setup_logger(log_path)
    logger.info(f"開始: source={source_root} dest={dest_root} excel={args.excel}")

    # dest 側に train/val/test を用意
    for subset in SUBSETS:
        (dest_root / subset).mkdir(parents=True, exist_ok=True)

    # Excel読込
    try:
        df = pd.read_excel(args.excel, sheet_name=args.sheet, engine="openpyxl")
    except Exception as e:
        logger.error(f"Excel読込エラー: {e}")
        raise SystemExit(1)

    if args.class_col not in df.columns:
        logger.error(f"指定のクラス名列 '{args.class_col}' が見つかりません。列名: {list(df.columns)}")
        raise SystemExit(1)

    # result列が無ければ作成
    if args.result_col not in df.columns:
        df[args.result_col] = ""

    # 行ごとに処理
    results = []
    for idx, row in df.iterrows():
        class_name = row[args.class_col]
        counts = process_row(class_name, source_root, dest_root, logger)
        # "train=10, val=2, test=0" 形式で記録
        result_text = f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
        df.at[idx, args.result_col] = result_text
        results.append((class_name, counts))

    # Excel上書き保存
    try:
        with pd.ExcelWriter(args.excel, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, index=False, sheet_name=args.sheet if args.sheet else "Sheet1")
    except Exception as e:
        logger.error(f"Excel書き込みエラー: {e}")
        raise SystemExit(1)

    logger.info("完了しました。")

if __name__ == "__main__":
    main()

python copy_class_folders.py \
  --excel data.xlsx \
  --sheet Sheet1 \
  --class-col "class name" \
  --result-col "result" \
  --source "D:/dataset/source" \
  --dest "D:/dataset/dest" \
  --log "copy_classes.log"
