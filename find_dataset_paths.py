#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excelのsheet1!A列にあるクラス名（例: XXXXX_CCCCCCC_XXXXXXXXXX または XXXXXXXX_CCCCCCCCCm_XXXXXXXXXX）を読み取り、
サーバにある datasets/{train,val,test} をチェックして、クラス名ディレクトリのパスを out シートに出力します。

仕様:
- まず元のクラス名で {subset}/{class_name} が存在するか確認（subset は train/val/test）
- いずれも存在しない場合、クラス名の真ん中の要素（2番目の "_" で区切られたトークン）の末尾の "m" を1回だけ除去して再検索
- それでも見つからなければ、名前のみ（パスは空）で出力します
- 出力列: input_name, used_query, train_path, val_path, test_path, status

使い方（例）:
    python find_dataset_paths.py --excel /path/to/workbook.xlsx --base /path/to/datasets --sheet_in sheet1 --sheet_out out
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def remove_trailing_m_from_middle(name: str) -> str:
    """クラス名の真ん中トークン末尾にある 'm' を1回だけ除去した名前を返す。なければ元の名前を返す。"""
    parts = name.split("_")
    if len(parts) < 3:
        return name  # 想定外フォーマットはそのまま
    mid = parts[1]
    if mid.endswith("m"):
        parts[1] = mid[:-1]  # 末尾mを1回だけ削除
        return "_".join(parts)
    return name


def check_paths(base_dir: Path, class_name: str, subsets: List[str]) -> Dict[str, str]:
    """各 subset について base/subset/class_name が存在するかを確認して、存在すればそのパス、なければ空文字を返す。"""
    results: Dict[str, str] = {}
    for sb in subsets:
        candidate = base_dir / sb / class_name
        results[sb] = str(candidate) if candidate.exists() else ""
    return results


def main():
    parser = argparse.ArgumentParser(description="Excelのクラス名からdatasets内のパスを収集してoutシートに書き出します。")
    parser.add_argument("--excel", required=True, help="入力Excelファイルのパス（既存ワークブック）")
    parser.add_argument("--base", required=True, help="datasetsのベースディレクトリ（例: /srv/datasets または //server/share/datasets）")
    parser.add_argument("--sheet_in", default="sheet1", help="入力シート名（既定: sheet1）")
    parser.add_argument("--sheet_out", default="out", help="出力シート名（既定: out）")
    parser.add_argument("--col", default="A", help="クラス名が入っている列（既定: A）")
    parser.add_argument("--header", type=int, default=None, help="ヘッダ行の行番号（0始まり）。ヘッダ無しなら None（既定）")
    parser.add_argument("--start_row", type=int, default=0, help="読み取り開始の行番号（0始まり）。既定: 0（ワークシート先頭から）")
    parser.add_argument("--subsets", nargs="*", default=["train", "val", "test"], help="検索するサブセット（既定: train val test）")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    base_dir = Path(args.base)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excelが見つかりません: {excel_path}")
    if not base_dir.exists():
        raise FileNotFoundError(f"datasetsベースが見つかりません: {base_dir}")

    # Excel読込
    # 指定列のみを読み込むため usecols を使用。開始行を考慮してスライス。
    df_all = pd.read_excel(excel_path, sheet_name=args.sheet_in, header=args.header, usecols=args.col)
    # 列名を正規化
    df_all.columns = ["class_name"]
    # 開始行をスキップ
    if args.start_row > 0:
        df_all = df_all.iloc[args.start_row :].reset_index(drop=True)

    # 空/NaNを除去
    df_all = df_all.dropna(subset=["class_name"])
    df_all["class_name"] = df_all["class_name"].astype(str).str.strip()
    df_all = df_all[df_all["class_name"] != ""].reset_index(drop=True)

    records: List[Dict[str, str]] = []

    for cls in df_all["class_name"]:
        # 1) 元名で検索
        paths1 = check_paths(base_dir, cls, args.subsets)
        found1 = any(paths1[sb] for sb in args.subsets)

        used_query = cls
        status = "FOUND_ORIGINAL" if found1 else "NOT_FOUND_TRY_STRIP_M"

        paths_final = paths1

        # 2) 見つからなければ、真ん中のトークンの末尾 'm' を外した名前で再検索
        if not found1:
            stripped = remove_trailing_m_from_middle(cls)
            if stripped != cls:
                paths2 = check_paths(base_dir, stripped, args.subsets)
                found2 = any(paths2[sb] for sb in args.subsets)
                if found2:
                    used_query = stripped
                    status = "FOUND_STRIPPED_M"
                    paths_final = paths2
                else:
                    status = "NOT_FOUND_ALL"
            else:
                status = "NOT_FOUND_ALL"

        rec = {
            "input_name": cls,
            "used_query": used_query,
            **{f"{sb}_path": paths_final.get(sb, "") for sb in args.subsets},
            "status": status,
        }
        records.append(rec)

    out_df = pd.DataFrame.from_records(records, columns=["input_name", "used_query"] + [f"{sb}_path" for sb in args.subsets] + ["status"])

    # Excelの out シートに書き出し（既存ファイルを更新）
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        out_df.to_excel(writer, sheet_name=args.sheet_out, index=False)

    print(f"Wrote {len(out_df)} rows to sheet '{args.sheet_out}' in {excel_path}")


if __name__ == "__main__":
    main()




python find_dataset_paths.py \
  --excel /path/to/workbook.xlsx \
  --base /srv/datasets \
  --sheet_in sheet1 \
  --sheet_out out
