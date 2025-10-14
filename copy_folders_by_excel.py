# -*- coding: utf-8 -*-
"""
Excelを元にクラスフォルダをコピーしてOK列を更新するスクリプト

使い方例:
python copy_by_excel.py \
  --excel_path data.xlsx \
  --sheet_name Sheet1 \
  --src_new_base "D:/dataset_new_source" \
  --dst_new_base "E:/dataset_new_target" \
  --src_hold_base "D:/dataset_hold_source" \
  --dst_hold_base "E:/dataset_hold_target"
"""

import argparse
import os
import shutil
from pathlib import Path
import pandas as pd

# --- 列名（必要に応じて変更可） ---
TEST_COL = "テスト列"          # 値は「新規」または「保留」
CLASS_COL = "新ルールクラス名"
OK_COL = "OK列"

SPLITS = ["train", "val", "test"]

def copy_class_folders(src_base: Path, dst_base: Path, class_name: str) -> dict:
    """
    src_base 直下の train/val/test から class_name フォルダを探して
    dst_base 直下にコピーする。存在する分割のみコピー。
    返り値: {"found": int, "copied": int, "errors": list[str]}
    """
    results = {"found": 0, "copied": 0, "errors": []}

    for split in SPLITS:
        src_dir = src_base / split / class_name
        if not src_dir.exists():
            continue  # 見つからない分割はスキップ
        results["found"] += 1

        dst_dir = dst_base / split / class_name
        try:
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            # 既存を上書きマージしたい場合は copytree(dirs_exist_ok=True)
            # 完全に置き換えたい場合は事前に削除
            if dst_dir.exists():
                # 既存がある場合は上書きマージ（ファイル単位）
                # shutil.copytree(..., dirs_exist_ok=True) で足りるが、
                # 既存を活かしつつ新規をコピーするため以下の方法でもOK
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            else:
                shutil.copytree(src_dir, dst_dir)
            results["copied"] += 1
        except Exception as e:
            results["errors"].append(f"{split}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", required=True, help="入力Excelファイルのパス（.xlsx 推奨）")
    parser.add_argument("--sheet_name", default=None, help="シート名（未指定なら先頭シート）")
    parser.add_argument("--src_new_base", required=True, help="新規のソース基底パス（直下に train/val/test）")
    parser.add_argument("--dst_new_base", required=True, help="新規のコピー先基底パス（直下に train/val/test）")
    parser.add_argument("--src_hold_base", required=True, help="保留のソース基底パス（直下に train/val/test）")
    parser.add_argument("--dst_hold_base", required=True, help="保留のコピー先基底パス（直下に train/val/test）")
    parser.add_argument("--backup_excel", action="store_true", help="Excelを上書き前に .bak を作成")
    args = parser.parse_args()

    excel_path = Path(args.excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excelが見つかりません: {excel_path}")

    # Excel 読み込み
    df = pd.read_excel(excel_path, sheet_name=args.sheet_name)

    # 必須列チェック
    missing_cols = [c for c in [TEST_COL, CLASS_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Excelに必須列がありません: {missing_cols}")

    # OK列が無ければ作る
    if OK_COL not in df.columns:
        df[OK_COL] = ""

    src_new_base = Path(args.src_new_base)
    dst_new_base = Path(args.dst_new_base)
    src_hold_base = Path(args.src_hold_base)
    dst_hold_base = Path(args.dst_hold_base)

    # ログ用
    total_rows = len(df)
    ok_count = 0
    ng_rows = []

    for idx, row in df.iterrows():
        status = str(row[TEST_COL]).strip() if pd.notna(row[TEST_COL]) else ""
        class_name = str(row[CLASS_COL]).strip() if pd.notna(row[CLASS_COL]) else ""

        if not class_name:
            ng_rows.append((idx, "クラス名が空"))
            continue

        if status == "新規":
            src_base = src_new_base
            dst_base = dst_new_base
        elif status == "保留":
            src_base = src_hold_base
            dst_base = dst_hold_base
        else:
            ng_rows.append((idx, f"不明なテスト列の値: {status!r}（想定: '新規' or '保留'）"))
            continue

        res = copy_class_folders(src_base, dst_base, class_name)

        # 成功判定: 少なくとも1分割以上見つかり、エラーが0
        if res["found"] >= 1 and len(res["errors"]) == 0:
            df.at[idx, OK_COL] = "OK"
            ok_count += 1
        else:
            reason = []
            if res["found"] == 0:
                reason.append("train/val/test のいずれにもクラスが存在しない")
            if res["errors"]:
                reason.append(" / ".join(res["errors"]))
            ng_rows.append((idx, " ; ".join(reason) if reason else "理由不明"))

        # 進捗表示（任意）
        print(f"[{idx+1}/{total_rows}] {class_name} -> found:{res['found']} copied:{res['copied']} errors:{len(res['errors'])}")

    # Excel バックアップ
    if args.backup_excel:
        bak = excel_path.with_suffix(excel_path.suffix + ".bak")
        shutil.copy2(excel_path, bak)
        print(f"バックアップ作成: {bak}")

    # Excel 上書き保存
    df.to_excel(excel_path, index=False)
    print(f"完了: {ok_count}/{total_rows} 行で OK を記載しました。")
    if ng_rows:
        print("未OK行の概要:")
        for i, msg in ng_rows:
            print(f"  行 {i+2}（ヘッダーを1行と仮定）: {msg}")

if __name__ == "__main__":
    main()
