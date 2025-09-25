#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ========== ユーティリティ ==========
def col_letter_to_index(letter: str) -> int:
    """Excelの列記号(A,B,...,Z,AA,AB,...) → 0始まりインデックス"""
    letter = letter.strip().upper()
    idx = 0
    for ch in letter:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid column letter: {letter}")
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1

def list_images(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files

def read_label_file(label_txt: Path) -> List[Tuple[str, str, str]]:
    """
    label_neta_fair_330.txt の各行:
      0001_123_まぐろ
      ^^^^ ^^^ 商品名
    戻り値: [(xxx, yyy, name), ...]  (xxx='0001', yyy='123', name='まぐろ')
    """
    rows = []
    with open(label_txt, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 最初の2つの '_' で分割（商品名に '_' が入っても対応）
            parts = s.split("_", 2)
            if len(parts) < 3:
                # 形式が違う行はスキップ
                continue
            xxx, yyy, name = parts[0], parts[1], parts[2]
            rows.append((xxx, yyy, name))
    return rows

def build_label_lookup(rows: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    商品コード(yyy) -> [(xxx, name), ...]  の辞書に。
    """
    lut: Dict[str, List[Tuple[str, str]]] = {}
    for xxx, yyy, name in rows:
        lut.setdefault(yyy, []).append((xxx, name))
    return lut

def normalize_text(s: str) -> str:
    return str(s).strip()

# ========== 主要ロジック ==========
def find_filename_in_excel_row(row: pd.Series, file_cols_idx: List[int], target_name: str) -> Optional[int]:
    """
    1行の中で、ファイル名1..8のどの列に一致があるか探す。
    見つかったら、その列のindex（0始まり, pandasの列番号）を返す。なければ None。
    """
    for idx in file_cols_idx:
        val = row.iloc[idx]
        if pd.isna(val):
            continue
        if normalize_text(val) == target_name:
            return idx
    return None

def count_nonempty_filecols(row: pd.Series, file_cols_idx: List[int]) -> int:
    """ファイル名1..8のうち非空セルの個数を数える。"""
    c = 0
    for idx in file_cols_idx:
        val = row.iloc[idx]
        if pd.notna(val) and normalize_text(val) != "":
            c += 1
    return c

def get_group_start(df: pd.DataFrame, current_row: int, group_col_idx: int) -> int:
    """
    現在行から上に向かって、A列（group_col_idx）に値がある最初の行を探す。
    それがグループ開始行。
    """
    r = current_row
    while r >= 0:
        v = df.iat[r, group_col_idx]
        if pd.notna(v) and str(v).strip() != "":
            return r
        r -= 1
    return 0  # 見つからなければ先頭をグループ開始とみなす

def check_group_consistency(df: pd.DataFrame, start_row: int, K: int, group_col_idx: int) -> bool:
    """
    グループ整合性チェック:
    - 開始行(start_row)のみA列に値、それ以外(start_row+1 .. start_row+K-1)はA列が空、を満たすか。
    - 行数不足の場合はFalse。
    """
    end_row = start_row + K - 1
    if end_row >= len(df):
        return False
    # 開始行は値あり
    v0 = df.iat[start_row, group_col_idx]
    if pd.isna(v0) or str(v0).strip() == "":
        return False
    # 残りは空
    for r in range(start_row + 1, end_row + 1):
        v = df.iat[r, group_col_idx]
        if pd.notna(v) and str(v).strip() != "":
            return False
    return True

def decide_true_label_by_code_and_name(
    product_code: str,
    product_name: str,
    label_lut: Dict[str, List[Tuple[str, str]]]
) -> Tuple[str, str]:
    """
    ラベルファイルから true_label / true_label_confirming を決める。
    - 商品コード一致かつ商品名一致 → true_label = "xxx_yyy_name"
    - 商品コード一致だが商品名不一致 → true_label_confirming に "xxx_yyy_name"（候補）を入れる
      （候補が複数ある場合は最初の候補を入れる。必要に応じて拡張可）
    - 商品コード一致がなければ、両方空
    """
    product_code = normalize_text(product_code)
    product_name = normalize_text(product_name)

    if product_code in label_lut:
        candidates = label_lut[product_code]  # [(xxx, name), ...]
        # 完全一致優先
        for xxx, name in candidates:
            if normalize_text(name) == product_name:
                return f"{xxx}_{product_code}_{name}", ""  # true_label, confirming
        # 一致なし → 確認用
        if candidates:
            xxx, name = candidates[0]
            return "", f"{xxx}_{product_code}_{name}"
    return "", ""

def ensure_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def process(
    miscls_root: Path,
    excel_path: Path,
    label_txt: Path,
    out_excel: Path,
    out_copy_root: Optional[Path],
    dry_run: bool,
    # 列定義（既定はご提示どおり）
    col_group_letter: str = "A",
    col_code_letter: str = "F",
    col_name_letter: str = "H",
    file_cols_letters: List[str] = ["J","L","N","P","R","T","V","X"],
):
    # --- 入力チェック ---
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")
    if not label_txt.exists():
        raise FileNotFoundError(f"Label text not found: {label_txt}")
    if not miscls_root.exists():
        raise FileNotFoundError(f"Misrecognized root not found: {miscls_root}")

    # --- Excel 読み込み（列記号でアクセスしたいので header=None で読み込む） ---
    df = pd.read_excel(excel_path, header=None)
    n_rows, n_cols = df.shape

    group_idx = col_letter_to_index(col_group_letter)
    code_idx = col_letter_to_index(col_code_letter)
    name_idx = col_letter_to_index(col_name_letter)
    file_cols_idx = [col_letter_to_index(c) for c in file_cols_letters]

    # --- ラベル辞書 ---
    label_rows = read_label_file(label_txt)
    label_lut = build_label_lookup(label_rows)

    # --- Excel内の「ファイル名 → (行idx, どの列idx)」逆引き索引を作る（高速化） ---
    filename_to_positions: Dict[str, List[Tuple[int,int]]] = {}
    for r in range(n_rows):
        for c in file_cols_idx:
            val = df.iat[r, c]
            if pd.isna(val):
                continue
            name = normalize_text(val)
            if not name:
                continue
            filename_to_positions.setdefault(name, []).append((r, c))

    # --- 誤認識画像の走査 ---
    results = []  # 出力行
    img_paths = list_images(miscls_root)

    for img_path in img_paths:
        filename = img_path.name
        relpath = str(img_path.relative_to(miscls_root))
        matches = filename_to_positions.get(filename, [])

        # デフォルト値
        true_label = ""
        true_label_conf = ""
        auto = "NG"
        manual = "要確認"
        reason = ""

        if not matches:
            # Excelに該当なし → 手動
            reason = "Excel未ヒット"
        elif len(matches) > 1:
            # 複数ヒット → 自動不可
            reason = f"Excel複数ヒット({len(matches)})"
        else:
            r, c = matches[0]
            # 商品コード・商品名取得
            product_code = df.iat[r, code_idx]
            product_name = df.iat[r, name_idx]
            product_code = "" if pd.isna(product_code) else str(product_code).strip()
            product_name = "" if pd.isna(product_name) else str(product_name).strip()

            # ラベル決定（コード優先、名前一致ならtrue_label、名前不一致ならconfirming）
            tl, tlc = decide_true_label_by_code_and_name(product_code, product_name, label_lut)
            true_label, true_label_conf = tl, tlc

            # 自動/手動判定
            if c == file_cols_idx[0]:  # ファイル名1(J列)にヒット
                if true_label:
                    auto, manual = "OK", ""
                else:
                    auto, manual = "NG", "要確認(名前不一致)"
            else:
                # ファイル名2〜8 の場合：グループ整合性チェック
                K = count_nonempty_filecols(df.iloc[r, :], file_cols_idx)
                start_row = get_group_start(df, r, group_idx)
                ok_group = check_group_consistency(df, start_row, K, group_idx)
                if not ok_group:
                    reason = f"グループ不整合(K={K}, start={start_row}, r={r})"
                    auto, manual = "NG", "要確認(グループ不整合)"
                else:
                    # グループOKなら、名前一致していれば自動OK
                    if true_label:
                        auto, manual = "OK", ""
                    else:
                        auto, manual = "NG", "要確認(名前不一致)"

        # コピー（dry_runでなければ、自動OKかつtrue_labelがある時のみ）
        if out_copy_root and (not dry_run) and auto == "OK" and true_label:
            dst_dir = out_copy_root / true_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / filename
            try:
                shutil.copy2(img_path, dst)
            except Exception as e:
                # コピー失敗は自動不可に落とし、手動扱いに
                auto, manual = "NG", f"コピー失敗: {e}"

        results.append({
            "誤認識画像パス": relpath,
            "画像名": filename,
            "true_label": true_label,
            "true_label_confirming": true_label_conf,
            "自動振り分け": auto,
            "手動": manual,
        })

    # --- 出力Excel ---
    out_df = pd.DataFrame(results, columns=["誤認識画像パス","画像名","true_label","true_label_confirming","自動振り分け","手動"])
    out_df.to_excel(out_excel, index=False)
    print(f"[DONE] 出力: {out_excel}  行数: {len(out_df)}")
    if out_copy_root:
        print(f"[INFO] コピー先: {out_copy_root}  dry_run={dry_run}")




python match_and_dispatch.py \
  --miscls_root "annotation_images" \
  --excel "reference.xlsx" \
  --labels "label_neta_fair_330.txt" \
  --out_excel "search_results.xlsx" \
  --out_copy_root "sorted_out" \
  --dry_run

--col_group A --col_code F --col_name H --file_cols J L N P R T V X

