# -*- coding: utf-8 -*-
"""
Excelの「商品番号」列とサーバフォルダ名（例: 1234_ABC）のプレフィックスを突き合わせ、
一致フォルダ内のPNG枚数をカウントしてExcelに出力するスクリプト。

必要ライブラリ:
    pip install pandas openpyxl

使い方:
    1) 下の CONFIG を環境に合わせて編集
    2) python match_count_png.py などで実行
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ======== CONFIG（ここを書き換えて使ってください）========
INPUT_EXCEL = r"input.xlsx"        # 入力Excelファイル
SHEET_NAME = "Sheet1"              # 読み込みシート名
PRODUCT_COL = "商品番号"            # 商品番号が入っている列名
SERVER_ROOT = r"\\server\share\root"  # サーバのルートパス（直下に 1234_xxx フォルダが並ぶ想定）
OUTPUT_EXCEL = r"output.xlsx"      # 出力Excelファイル

# フォルダ内のPNGを数えるときにサブフォルダまで再帰するか
RECURSIVE = False  # 通常は直下のみカウント（要件「該当フォルダ中」から非再帰を想定）
# =======================================================


def normalize_code(raw) -> str:
    """
    Excelの元値から数字のみを抽出し、4桁ゼロ埋めに正規化。
    例: "12-34" -> "1234", "7" -> "0007"
    """
    if pd.isna(raw):
        return ""
    s = str(raw)
    digits = re.sub(r"\D", "", s)  # 数字以外除去
    if digits == "":
        return ""
    return digits.zfill(4)


def build_prefix_map(root: Path) -> Dict[str, List[Path]]:
    """
    サーバルート直下のサブフォルダから、フォルダ名の'_'より前の数字を取り出して
    4桁ゼロ埋めし、{prefix: [フォルダPath,...]} の辞書を作る。
    """
    mapping: Dict[str, List[Path]] = {}
    if not root.exists():
        raise FileNotFoundError(f"サーバルートが存在しません: {root}")

    for p in root.iterdir():
        if not p.is_dir():
            continue
        # フォルダ名の '_' より前を抽出
        name = p.name
        prefix = name.split("_", 1)[0]
        # 数字以外が混ざる可能性を考え、数字だけ抽出
        prefix_digits = re.sub(r"\D", "", prefix)
        if prefix_digits == "":
            continue
        key = prefix_digits.zfill(4)
        mapping.setdefault(key, []).append(p)
    return mapping


def count_pngs_in_folder(folder: Path, recursive: bool = False) -> int:
    """
    指定フォルダ配下のPNGファイル数をカウント。
    recursive=False: 直下のみ
    recursive=True : サブフォルダも含める
    """
    if not folder.exists():
        return 0
    if recursive:
        # サブフォルダも含めて数える
        count = 0
        for root, dirs, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(".png"):
                    count += 1
        return count
    else:
        # 直下のみ
        return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".png")


def process(
    input_excel: str,
    sheet_name: str,
    product_col: str,
    server_root: str,
    output_excel: str,
    recursive: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[Path]]]:
    """
    主処理：
      - Excel読み込み
      - 商品番号正規化（4桁）
      - サーバ側フォルダのプレフィックスマップ作成
      - 突合してPNG数カウント（同プレフィックス複数フォルダあれば合算）
      - 結果をDataFrameで返し、同時にExcel出力
    """
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    if product_col not in df.columns:
        raise KeyError(f"指定の列が見つかりません: {product_col}")

    # 商品番号正規化列を追加
    df["_正規化商品番号"] = df[product_col].apply(normalize_code)

    # サーバ側フォルダマップ作成
    root = Path(server_root)
    prefix_map = build_prefix_map(root)

    results = []
    for _, row in df.iterrows():
        raw_code = row[product_col]
        code = row["_正規化商品番号"]

        if not code:
            results.append(
                {
                    "商品番号": raw_code,
                    "フォルダパス": "",
                    "PNGファイル数": 0,
                }
            )
            continue

        folders = prefix_map.get(code, [])
        if not folders:
            results.append(
                {
                    "商品番号": raw_code,
                    "フォルダパス": "",
                    "PNGファイル数": 0,
                }
            )
            continue

        # 同一プレフィックスのフォルダが複数ある場合はパスを;区切りで出力し、PNG数は合計
        paths_str = ";".join(str(p) for p in folders)
        total_png = sum(count_pngs_in_folder(p, recursive=recursive) for p in folders)

        results.append(
            {
                "商品番号": raw_code,
                "フォルダパス": paths_str,
                "PNGファイル数": total_png,
            }
        )

    out_df = pd.DataFrame(results, columns=["商品番号", "フォルダパス", "PNGファイル数"])

    # 出力：既存の他シートを保持したい場合は ExcelWriter(mode="a") を検討
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="結果", index=False)

    return out_df, prefix_map


if __name__ == "__main__":
    out_df, _ = process(
        INPUT_EXCEL,
        SHEET_NAME,
        PRODUCT_COL,
        SERVER_ROOT,
        OUTPUT_EXCEL,
        recursive=RECURSIVE,
    )
    print("出力件数:", len(out_df))
    print("完了:", OUTPUT_EXCEL)
