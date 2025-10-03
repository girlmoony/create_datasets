#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Excelの指定シートを読み、テスト列が「保留」「新規」「削除→保留」の行について
SERVER_ROOT/{train,test,val}/{クラス名} を調査し、
存在するフォルダのパスと直下（または再帰）ファイル枚数を
同一シート構造で新規Excel ( *_updated.xlsx ) に書き出します。
"""

from pathlib import Path
import argparse
import sys
import pandas as pd

# ====== 設定（必要に応じて変更） ==========================================
TARGET_STATUSES = {"保留", "新規", "削除→保留"}  # 対象とする「テスト列」の値
SPLITS = ["train", "test", "val"]               # 探索する分割フォルダ
COL_STATUS = "テスト列"                          # ステータス列名
COL_CLASS = "新しいクラス名"                     # クラス名（フォルダ名）列名
COL_OUT_PATH = "結果パス"                        # 出力：ヒットしたフォルダのフルパス（複数可）
COL_OUT_COUNT = "結果枚数"                       # 出力：対応する枚数（「; 」区切りで複数）
# =======================================================================

def count_files_in_dir(dir_path: Path, recursive: bool) -> int:
    """ディレクトリ内ファイル枚数を数える（recursive=False: 直下のみ, True: 再帰）"""
    if not dir_path.is_dir():
        return 0
    if recursive:
        return sum(1 for p in dir_path.rglob("*") if p.is_file())
    else:
        return sum(1 for p in dir_path.iterdir() if p.is_file())

def process_row(server_root: Path, cls_name: str, recursive: bool) -> tuple[str, str]:
    """
    1行（クラス名）について train/test/val を探索し、
    見つかったフォルダのパスと枚数を「; 」区切りで返す。
    見つからなければ空文字を返す。
    """
    hit_paths: list[str] = []
    counts: list[str] = []
    for split in SPLITS:
        dir_path = server_root / split / cls_name
        if dir_path.is_dir():
            hit_paths.append(str(dir_path))
            counts.append(str(count_files_in_dir(dir_path, recursive)))
    return ("; ".join(hit_paths), "; ".join(counts))

def main():
    parser = argparse.ArgumentParser(description="Excelのクラスフォルダ枚数カウント（dtype/CRC対応版）")
    parser.add_argument("excel_path", help="入力Excelファイルのパス（読み取り専用・新規ファイルに保存）")
    parser.add_argument("--sheet", default=0, help="対象シート名またはインデックス（デフォルト: 0）")
    parser.add_argument("--server-root", required=True, help="サーバのルートパス（例: /data/dataset）")
    parser.add_argument("--recursive", action="store_true", help="サブフォルダも含めて枚数をカウント")
    parser.add_argument("--out", default=None, help="出力先Excelパス（省略時は *_updated.xlsx）")
    args = parser.parse_args()

    excel_path = Path(args.excel_path)
    server_root = Path(args.server_root)

    if not excel_path.exists():
        print(f"[ERROR] Excelが見つかりません: {excel_path}", file=sys.stderr)
        sys.exit(1)
    if not server_root.exists():
        print(f"[ERROR] サーバルートが見つかりません: {server_root}", file=sys.stderr)
        sys.exit(1)

    # シート読み込み（openpyxlで読込のみ）
    try:
        df = pd.read_excel(excel_path, sheet_name=args.sheet, engine="openpyxl")
    except Exception as e:
        print(f"[ERROR] Excel読み込み失敗: {e}", file=sys.stderr)
        sys.exit(1)

    # 必須列チェック
    for col in (COL_STATUS, COL_CLASS):
        if col not in df.columns:
            print(f"[ERROR] 必須列が見つかりません: {col}", file=sys.stderr)
            sys.exit(1)

    # 列型の整備：クラス名は文字列化・トリム
    df[COL_CLASS] = df[COL_CLASS].astype("string").str.strip()

    # 出力列をstring dtypeで準備（既存があれば型揃え）
    for col in (COL_OUT_PATH, COL_OUT_COUNT):
        if col not in df.columns:
            df[col] = pd.Series("", index=df.index, dtype="string")
        else:
            df[col] = df[col].astype("string")

    # 対象行のマスク
    status_series = df[COL_STATUS].astype("string").str.strip()
    mask = status_series.isin(TARGET_STATUSES)

    # 行ごとに処理
    for idx in df[mask].index:
        cls = df.at[idx, COL_CLASS]
        if cls is None or pd.isna(cls) or str(cls).strip() == "":
            df.at[idx, COL_OUT_PATH] = ""
            df.at[idx, COL_OUT_COUNT] = ""
            continue
        out_path, out_count = process_row(server_root, str(cls), args.recursive)
        # string dtype へ安全に代入
        df.at[idx, COL_OUT_PATH]  = (out_path or "")
        df.at[idx, COL_OUT_COUNT] = (out_count or "")

    # ===== 書き出し：壊れた埋め込み画像によるCRCエラー回避のため、新規ブックに保存 =====
    if args.out:
        out_xlsx = Path(args.out)
    else:
        out_xlsx = excel_path.with_name(excel_path.stem + "_updated.xlsx")

    # シート名確定（文字列で渡すことを推奨）
    sheet_name = args.sheet if isinstance(args.sheet, str) else "Sheet1"

    try:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"[OK] 新しいファイルに書き出しました: {out_xlsx}")
    except Exception as e:
        print(f"[ERROR] Excel書き出し失敗: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


# 直下のファイルのみカウント、Sheet1 を対象、出力は *_updated.xlsx
python check_server_datasets_fixed.py "C:\path\to\file.xlsx" --sheet "Sheet1" --server-root "\\server\dataset"

# サブフォルダも含めてカウント（--recursive）
python check_server_datasets_fixed.py "C:\path\to\file.xlsx" --sheet 0 --server-root "\\server\dataset" --recursive

# 出力先を明示
python check_server_datasets_fixed.py "C:\path\to\file.xlsx" --sheet "Sheet1" --server-root "\\server\dataset" --out "C:\path\to\file_updated.xlsx"

