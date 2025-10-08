# save as make_camera_summary.py
import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# [..][..] が任意個 → 店舗コード → 何らか → -seat- を拾う
# 例: [0.46][1.00]1289_110_10-08-00_2025-09-11_....png
FNAME_RE = re.compile(r"""
    ^(?:\[[^\]]+\])*         # 先頭の [..] 群をスキップ（0回以上）
    (?P<store>\d+)           # 店舗コード（連続数字）
    .*?                      # 適当な区切り
    -(?P<seat>\d{2})-        # ハイフンで囲まれた2桁seat
""", re.VERBOSE)

def iter_class_images(root_dirs):
    """各ルート直下のクラスフォルダ配下の画像パスを (class_name, path) でyield"""
    for root in root_dirs:
        root = Path(root)
        if not root.is_dir():
            continue
        for cls_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            cls = cls_dir.name
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    yield cls, p

def load_store_camera_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # 正規化（seatキーは "01"～"12" で来る想定）
    return meta

def make_camera_summary(root_dirs, json_path):
    meta = load_store_camera_json(json_path)
    # 集計キー: (class, store_code)
    # 値: {"型": str or None, "上": int, "下": int}
    agg = defaultdict(lambda: {"型": None, "上": 0, "下": 0})

    skipped = 0
    for cls, img_path in iter_class_images(root_dirs):
        m = FNAME_RE.search(img_path.name)
        if not m:
            skipped += 1
            continue
        store = m.group("store")          # '1289'
        seat = m.group("seat")            # '08'

        # 店舗メタ取得
        store_meta = meta.get(store)
        if not store_meta:
            skipped += 1
            continue

        # 型（新型/旧型）
        model_type = store_meta.get("型", None)
        # seat→上下
        seat_meta = store_meta.get(seat)
        if not seat_meta:
            skipped += 1
            continue
        updown = seat_meta.get("上下")  # "上段" or "下段"
        if updown not in ("上段", "下段"):
            skipped += 1
            continue

        key = (cls, store)
        if agg[key]["型"] is None:
            agg[key]["型"] = model_type
        # 「左右」は合算せず、上下だけカウント
        if updown == "上段":
            agg[key]["上"] += 1
        else:
            agg[key]["下"] += 1

    # DataFrameへ
    rows = []
    for (cls, store), v in agg.items():
        rows.append({
            "クラス名": cls,
            "店舗コード": store,
            "新型/旧型": v["型"],
            "カメラ位置（上）": v["上"],
            "カメラ位置（下）": v["下"],
        })
    df = pd.DataFrame(rows).sort_values(["クラス名", "店舗コード"])
    return df, skipped

if __name__ == "__main__":
    # 使い方例
    roots = [
        "/data/dataset_v1",
        "/data/dataset_extra",
    ]
    json_path = "stores_and_cameras.json"  # ご提示のJSON

    df, skipped = make_camera_summary(roots, json_path)
    out_xlsx = "camera_summary.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="camera_summary", index=False)
    print(f"書き出し: {out_xlsx}（パース不可のファイル {skipped} 枚）")
