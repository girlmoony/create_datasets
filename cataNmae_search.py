# -*- coding: utf-8 -*-
"""
Excelの「商品名」列をサーバ上のフォルダ名と比較し、該当フォルダ中のPNG数を集計して
Excelに出力するスクリプト。

仕様：
- フォルダ名は2種類：
    1) "XXXX"（アンダースコアなし） → フォルダ名全体と商品名を比較
    2) "XXXX_XXX"（アンダースコアあり） → "_"以降（XXX）と商品名を比較
- サーバ配下は再帰的に探索（フォルダの発見は再帰）
- PNG枚数はデフォルト「直下のみ」カウント（再帰カウントに切り替え可能）
- 出力列：商品名 / フォルダパス１（複数は;区切り） / ファイル数１（;区切りでフォルダ対応順）

前提：
    pip install pandas openpyxl
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ======== CONFIG（環境に合わせて修正）========
INPUT_EXCEL      = r"input.xlsx"       # 読み取り元Excel
SHEET_NAME       = "Sheet1"            # シート名
PRODUCT_COL      = "商品名"            # 商品名列の見出し
SERVER_ROOT      = r"\\server\share"   # サーバのルートフォルダ
OUTPUT_EXCEL     = r"output.xlsx"      # 出力Excel
RECURSIVE_COUNT  = False               # PNG枚数カウントを再帰にするなら True
IMAGE_EXTS       = {".png", ".PNG"}    # 対象拡張子
# ============================================


def normalize_text(s: str) -> str:
    """
    日本語を含む文字列の比較用に正規化：
    - Unicode正規化（NFKC）
    - 前後空白削除
    - 全空白類（半角/全角）を単一の空文字に
    - 小文字化
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    # すべての空白（全角スペース含む）を削除
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    return s


def build_folder_index(root: Path) -> Dict[str, List[Path]]:
    """
    サーバルート配下を再帰的に探索し、
    - 'XXXX'       → キー = normalize('XXXX')
    - 'XXXX_XXX'   → キー = normalize('XXX')
    のマッピング（キー→フォルダPathのリスト）を作成。
    """
    index: Dict[str, List[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        name = d.name
        # ルート自身（root）を除く。rootそのものを一致対象にしたいならこのifを削除
        if d == root:
            continue

        if "_" in name:
            # '_' 以降だけで比較
            suffix = name.split("_", 1)[1]
            key = normalize_text(suffix)
        else:
            key = normalize_text(name)

        if key:
            index.setdefault(key, []).append(d)

    return index


def count_pngs(folder: Path, recursive: bool = False) -> int:
    """
    PNGファイル数をカウント。
    recursive=False：直下のみ
    recursive=True ：サブフォルダも含めてカウント
    """
    if not folder.exists():
        return 0

    if not recursive:
        return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS)

    # 再帰カウント
    cnt = 0
    for dp, _, files in os.walk(folder):
        for fn in files:
            if Path(fn).suffix in IMAGE_EXTS:
                cnt += 1
    return cnt


def process(
    input_excel: str,
    sheet_name: str,
    product_col: str,
    server_root: str,
    output_excel: str,
    recursive_count: bool = False,
) -> pd.DataFrame:
    # Excel読み込み（商品名は文字列として扱う）
    df = pd.read_excel(input_excel, sheet_name=sheet_name, dtype={product_col: str})
    if product_col not in df.columns:
        raise KeyError(f"指定の列が見つかりません: {product_col}")

    # サーバ配下を再帰探索してインデックス化
    root = Path(server_root)
    if not root.exists():
        raise FileNotFoundError(f"サーバルートが存在しません: {root}")
    index = build_folder_index(root)

    # 行ごとに照合
    out_rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        prod_raw = row.get(product_col, "")
        prod_key = normalize_text(prod_raw)

        matches = index.get(prod_key, [])

        # パスとファイル数を同じ順で並べる
        paths: List[str] = []
        counts: List[str] = []
        for folder in matches:
            paths.append(str(folder))
            counts.append(str(count_pngs(folder, recursive=recursive_count)))

        out_rows.append(
            {
                "商品名": prod_raw,
                "フォルダパス1": ";".join(paths),
                "ファイル数1": ";".join(counts),
            }
        )

    out_df = pd.DataFrame(out_rows, columns=["商品名", "フォルダパス1", "ファイル数1"])

    # 出力（既存ブックに他シートがある場合は併用方法を調整）
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="結果", index=False)

    return out_df


if __name__ == "__main__":
    out = process(
        INPUT_EXCEL,
        SHEET_NAME,
        PRODUCT_COL,
        SERVER_ROOT,
        OUTPUT_EXCEL,
        recursive_count=RECURSIVE_COUNT,
    )
    print("出力件数:", len(out))
    print("完了:", OUTPUT_EXCEL)
