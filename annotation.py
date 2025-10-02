# -*- coding: utf-8 -*-
"""
Excelを1行ずつ処理し、各行のフォルダから PNG をランダムに選定して
（重複回避・暗い/ぼやけ除外・ファイル名で真上判定）、
コピー枚数に達するまでコピーする。

前提:
- Excelの対象シートに「フォルダパス」「コピー枚数」列がある
- フォルダパスは1つ想定（";"区切りで複数でも動作はしますが、まとめて候補にします）

依存:
    pip install pandas openpyxl opencv-python numpy piexif
"""

import os
import random
import shutil
from pathlib import Path
from typing import List, Iterable, Set

import numpy as np
import pandas as pd
import cv2
import piexif

# ========= 設定（環境に合わせて変更） =========
RESULT_EXCEL   = r"output.xlsx"
SHEET_NAME     = "結果"

FOLDER_COL     = "フォルダパス"   # 例: \\server\share\root\0123_ABC
COUNT_COL      = "コピー枚数"     # 例: 5（空/非数は0扱いにする）

LOCAL_ROOT     = r"D:\local\images"  # コピー先：同名フォルダをここに作る
RECURSIVE      = False               # サブフォルダも探索するなら True

# 判定しきい値（必要に応じて調整）
BLUR_THRESHOLD = 100.0     # Laplacian分散：これ未満はボケ
BRIGHTNESS_MIN = 40.0      # 平均輝度：これ未満は暗い（0-255）

# 「真上（トップダウン）」を示すキーワード（ファイル名に含まれていれば採用）
TOPDOWN_POS = [
    "top", "topdown", "overhead", "flatlay", "bird", "birdseye",
    "真上", "真俯瞰", "俯瞰", "天面", "_td", "-td"
]
# 真上ではなさそうなキーワード（含まれていたら除外）
TOPDOWN_NEG = [
    "front", "side", "angle", "angled", "persp", "perspective",
    "斜め", "横", "正面", "側面", "背面", "iso", "isometric"
]

IMAGE_EXTS    = {".png", ".PNG"}
WRITE_LOG_SHEET = True
LOG_SHEET_NAME  = "コピー実行ログ"
# ===========================================


def parse_copy_count(x) -> int:
    if x is None or str(x).strip() == "":
        return 0
    try:
        return max(0, int(float(str(x).strip())))
    except Exception:
        return 0


def list_images(folder: Path, recursive: bool = False) -> List[Path]:
    if not folder.exists():
        return []
    if recursive:
        out = []
        for root, _, files in os.walk(folder):
            for fn in files:
                p = Path(root) / fn
                if p.suffix in IMAGE_EXTS:
                    out.append(p)
        return out
    else:
        return [p for p in folder.iterdir() if p.is_file() and p.suffix in IMAGE_EXTS]


def is_topdown_by_name(p: Path) -> bool:
    name = p.stem.lower()
    if any(kw in name for kw in (k.lower() for k in TOPDOWN_NEG)):
        return False
    # 明示的なポジティブがあれば True、無ければ False（＝「真上」確証がないものは落とす）
    return any(kw in name for kw in (k.lower() for k in TOPDOWN_POS))


def is_topdown_by_exif_if_available(p: Path) -> bool:
    """PNGはEXIFがないことが多いので、あれば補助的に使う。無い場合はFalseを返し、名前判定に任せる。"""
    try:
        exif_dict = piexif.load(str(p))
    except Exception:
        return False
    # 説明/コメントにポジティブ語があれば True、ネガあれば False
    def b2s(b):
        if isinstance(b, bytes):
            try:
                return b.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        return str(b)

    desc = b2s(exif_dict.get("0th", {}).get(piexif.ImageIFD.ImageDescription, b"")).lower()
    ucom = b2s(exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")).lower()
    text = f"{desc} {ucom}"
    if any(kw.lower() in text for kw in TOPDOWN_NEG):
        return False
    if any(kw.lower() in text for kw in TOPDOWN_POS):
        return True
    return False


def laplacian_var(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(img_bgr) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def pass_quality_filters(p: Path) -> bool:
    """暗い/ボケを除外する"""
    # Windowsパス対策：imdecodeで読む
    img_bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return False
    if laplacian_var(img_bgr) < BLUR_THRESHOLD:
        return False
    if mean_brightness(img_bgr) < BRIGHTNESS_MIN:
        return False
    return True


def choose_random_images_until(
    candidates: List[Path],
    need: int,
    already_selected_names: Set[str],
) -> List[Path]:
    """
    候補からランダムに1枚ずつ取り出してチェックし、必要数に達するまで選ぶ。
    - 重複チェック：ファイル名で判定（同じ行内で同名を選ばない）
    - 真上判定：EXIF（あれば）OR ファイル名
    - 品質：暗い/ぼやけ除外
    """
    if need <= 0 or not candidates:
        return []

    pool = candidates[:]            # 元を壊さない
    random.shuffle(pool)            # ランダム順
    picked: List[Path] = []

    for p in pool:
        if len(picked) >= need:
            break

        # 重複（同じ行内で同名）を避ける
        if p.name in already_selected_names:
            continue

        # 真上チェック（EXIFにポジがあれば即OK、無ければ名前で判定）
        td = is_topdown_by_exif_if_available(p)
        if not td:
            # EXIFで決まらなければ名前で判定（ポジ必須）
            if not is_topdown_by_name(p):
                continue

        # 品質チェック
        if not pass_quality_filters(p):
            continue

        picked.append(p)
        already_selected_names.add(p.name)

        if len(picked) >= need:
            break

    return picked


def ensure_local_dir_for(server_folder_path: Path) -> Path:
    """
    サーバのフォルダ名と同名のローカルフォルダを LOCAL_ROOT に作る。
    （例：\\server\...\0123_ABC → D:\local\images\0123_ABC）
    """
    local_dir = Path(LOCAL_ROOT) / server_folder_path.name
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """
    同名があれば _1, _2 ... を付けて重複回避してコピー
    """
    dst = dst_dir / src.name
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = dst_dir / f"{stem}_{i}{suf}"
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.copy2(src, dst)
    return dst


def parse_folder_cell(cell_value) -> List[Path]:
    """
    セルが1つのパス想定だが、';'区切りで複数あってもOK（まとめて候補集め）
    """
    if cell_value is None:
        return []
    s = str(cell_value).strip()
    if not s:
        return []
    parts = [x.strip() for x in s.split(";") if x.strip()]
    return [Path(p) for p in parts]


def main():
    random.seed()
    df = pd.read_excel(RESULT_EXCEL, sheet_name=SHEET_NAME)

    # 必須列チェック（自動追加はしない）
    for col in (FOLDER_COL, COUNT_COL):
        if col not in df.columns:
            raise KeyError(f"対象シートに『{col}』列がありません。")

    logs = []

    # ---- Excelを1行ずつ処理 ----
    for idx, row in df.iterrows():
        folders = parse_folder_cell(row[FOLDER_COL])
        need = parse_copy_count(row[COUNT_COL])

        if not folders or need <= 0:
            continue

        # その行で既に選んだファイル名の集合（重複回避用）
        selected_names: Set[str] = set()

        # 候補の収集（1つのパス前提でも問題なし。複数なら全て合算）
        candidates: List[Path] = []
        for fol in folders:
            candidates.extend(list_images(fol, recursive=RECURSIVE))

        # ランダムに1枚ずつチェックしながら選ぶ（needに達するまで）
        picked = choose_random_images_until(candidates, need, selected_names)

        # コピー（ローカルは先頭フォルダ名で作成）
        local_dir = ensure_local_dir_for(folders[0])
        copied = []
        for p in picked:
            dst = safe_copy(p, local_dir)
            copied.append((p, dst))

        logs.append(
            f"[row {idx}] 要求:{need} / 選定:{len(picked)} / コピー先:{local_dir} "
            f"(元フォルダ数:{len(folders)}, 候補総数:{len(candidates)})"
        )

    # ログをExcelに
    if WRITE_LOG_SHEET:
        with pd.ExcelWriter(RESULT_EXCEL, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
            pd.DataFrame({"log": logs}).to_excel(w, sheet_name=LOG_SHEET_NAME, index=False)

    print("\n".join(logs))
    print("完了")


if __name__ == "__main__":
    main()
