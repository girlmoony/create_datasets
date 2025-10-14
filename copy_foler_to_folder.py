# -*- coding: utf-8 -*-
"""
AAA(元) 直下のフォルダにある画像を、BBB(先) 直下の同名フォルダにコピーする。
同名ファイルが先に存在する場合は上書きしない。
コピー結果をExcelに「コピー先 / コピー元 / コピー成功」で記録する。

使い方例:
python copy_images_no_overwrite.py \
  --src_base "D:/AAA" \
  --dst_base "E:/BBB" \
  --log_excel "copy_log.xlsx"
"""

import argparse
from pathlib import Path
import shutil
import sys
import pandas as pd

# 対象とする画像拡張子（小文字で比較）
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_base", required=True, help="元(AAA)のパス。直下にクラス/ID等のフォルダが並んでいることを想定")
    parser.add_argument("--dst_base", required=True, help="先(BBB)のパス。直下に同名フォルダを作成/利用してコピー")
    parser.add_argument("--log_excel", required=True, help="結果ログのExcelファイルの出力パス")
    parser.add_argument("--make_missing_dir", action="store_true",
                        help="先(BBB)側に同名フォルダが無い場合は新規作成（デフォルトは作成する）")
    args = parser.parse_args()

    src_base = Path(args.src_base)
    dst_base = Path(args.dst_base)
    log_path = Path(args.log_excel)
    make_missing = True if not args.make_missing_dir else True  # 互換のため常にTrue

    if not src_base.exists() or not src_base.is_dir():
        print(f"元(AAA)が見つからないかディレクトリではありません: {src_base}", file=sys.stderr)
        sys.exit(1)
    if not dst_base.exists():
        try:
            dst_base.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"先(BBB)を作成できませんでした: {dst_base} / {e}", file=sys.stderr)
            sys.exit(1)

    logs = []  # 各ファイルごとにログを残す

    # 「AAA直下のフォルダ一覧中の画像」が対象：→ AAA直下の1階層目のフォルダのみ走査
    for sub in sorted([p for p in src_base.iterdir() if p.is_dir()]):
        src_folder = sub
        dst_folder = dst_base / sub.name

        # 先側フォルダが無ければ作成（既存の画像に影響なし）
        if not dst_folder.exists():
            try:
                dst_folder.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"先フォルダ作成に失敗: {dst_folder} / {e}", file=sys.stderr)
                continue

        # サブフォルダ直下のファイルだけを見る（再帰しない仕様）
        for item in sorted(src_folder.iterdir()):
            if not is_image(item):
                continue

            dst_file = dst_folder / item.name

            if dst_file.exists():
                # 既存あり → 上書きしない
                logs.append({
                    "コピー先": str(dst_file),
                    "コピー元": str(item),
                    "コピー成功": False  # スキップ
                })
                continue

            try:
                # コピー（メタデータも含める）
                shutil.copy2(item, dst_file)
                logs.append({
                    "コピー先": str(dst_file),
                    "コピー元": str(item),
                    "コピー成功": True
                })
            except Exception as e:
                print(f"コピー失敗: {item} -> {dst_file} / {e}", file=sys.stderr)
                logs.append({
                    "コピー先": str(dst_file),
                    "コピー元": str(item),
                    "コピー成功": False
                })

    # ログをExcel出力
    df = pd.DataFrame(logs, columns=["コピー先", "コピー元", "コピー成功"])
    # 上書き保存（既存ファイルがあれば置き換える）
    df.to_excel(log_path, index=False)
    print(f"完了: ログを出力しました -> {log_path}")
    print(f"総対象ファイル数: {len(df)} / コピー成功: {df['コピー成功'].sum()} / スキップ(既存等): {(~df['コピー成功']).sum()}")

if __name__ == "__main__":
    main()
