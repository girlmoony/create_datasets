#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Excelの指定シートを読み、テスト列が「保留」「新規」「削除→保留」の行について
SERVER_ROOT/{train,test,val}/{クラス名} を調査し、
存在するフォルダのパスと直下のファイル枚数を Excel の同じシートに書き戻します。
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

def count_files_in_dir(dir_path: Path) -> int:
    """ディレクトリ直下のファイル枚数を数える（サブディレクトリは無視）"""
    if not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file())

def process_row(server_root: Path, cls_name: str) -> tuple[str, str]:
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
            counts.append(str(count_files_in_dir(dir_path)))
    return ("; ".join(hit_paths), "; ".join(counts))

def main():
    parser = argparse.ArgumentParser(description="Excelのクラスフォルダ枚数カウント書き戻しツール")
    parser.add_argument("excel_path", help="入力Excelファイルのパス（上書き保存されます）")
    parser.add_argument("--sheet", default=0,
                        help="対象シート名またはインデックス（デフォルト: 0）")
    parser.add_argument("--server-root", required=True,
                        help="サーバのルートパス（例: /data/dataset）")
    parser.add_argument("--backup", action="store_true",
                        help="Excelを上書きする前に .bak を作る")
    args = parser.parse_args()

    excel_path = Path(args.excel_path)
    server_root = Path(args.server_root)

    if not excel_path.exists():
        print(f"[ERROR] Excelが見つかりません: {excel_path}", file=sys.stderr)
        sys.exit(1)
    if not server_root.exists():
        print(f"[ERROR] サーバルートが見つかりません: {server_root}", file=sys.stderr)
        sys.exit(1)

    # バックアップ
    if args.backup:
        bak = excel_path.with_suffix(excel_path.suffix + ".bak")
        bak.write_bytes(excel_path.read_bytes())
        print(f"[INFO] バックアップ作成: {bak}")

    # シート読み込み
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

    # 出力列が無ければ作る
    if COL_OUT_PATH not in df.columns:
        df[COL_OUT_PATH] = ""
    if COL_OUT_COUNT not in df.columns:
        df[COL_OUT_COUNT] = ""

    # 前処理：クラス名を文字列化・トリム
    df[COL_CLASS] = df[COL_CLASS].astype(str).str.strip()

    # 対象行のマスク
    status_series = df[COL_STATUS].astype(str).str.strip()
    mask = status_series.isin(TARGET_STATUSES)

    # 行ごとに処理
    for idx in df[mask].index:
        cls = df.at[idx, COL_CLASS]
        if not cls or cls.lower() == "nan":
            df.at[idx, COL_OUT_PATH] = ""
            df.at[idx, COL_OUT_COUNT] = ""
            continue
        out_path, out_count = process_row(server_root, cls)
        df.at[idx, COL_OUT_PATH] = out_path
        df.at[idx, COL_OUT_COUNT] = out_count

    # 書き戻し（同じシートを上書き）
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            sheet_name = args.sheet if isinstance(args.sheet, str) else writer.book.sheetnames[args.sheet]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        print(f"[OK] 書き込み完了: {excel_path}")
    except Exception as e:
        print(f"[ERROR] Excel書き込み失敗: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
