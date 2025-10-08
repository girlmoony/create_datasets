#!/usr/bin/env python3
# save as curate_dataset.py
# usage:
#   python curate_dataset.py \
#     --ok-root /data/ok_root \
#     --wrong-root /data/wrong_root \
#     --base-root /data/base_root \
#     --store-json /data/stores_and_cameras.json \
#     --excel-classes /lists/classes.xlsx --sheet summary --col "class name" \
#     --dest /data/curated_out \
#     --train 140 --val 30 --test 30 \
#     --excel summary.xlsx \
#     --dry-run
#
# 必要: pandas, openpyxl

import argparse
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
import shutil
import random
import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 例: [0.46][1.00]1289_110_10-08-00_2025-09-11_19-22-55-044_1.png
RE_SCORE2 = re.compile(r"^\[[^\]]*\]\[(?P<score>\d+(?:\.\d+)?)\]")
RE_STORE = re.compile(r"(?:\[[^\]]+\])*?(?P<store>\d+)")
RE_SEAT = re.compile(r"-(?P<seat>\d{2})-")

def parse_score2(name: str) -> Optional[float]:
    m = RE_SCORE2.search(name)
    if not m:
        return None
    try:
        return float(m.group("score"))
    except:
        return None

def parse_store_and_seat(name: str) -> Tuple[Optional[str], Optional[str]]:
    m1 = RE_STORE.search(name)
    store = m1.group("store") if m1 else None
    m2 = RE_SEAT.search(name)
    seat = m2.group("seat") if m2 else None
    return store, seat

def load_store_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_updown(store_meta: Dict, store: Optional[str], seat: Optional[str]) -> Optional[str]:
    if not store or not seat:
        return None
    info = store_meta.get(store)
    if not info:
        return None
    seat_info = info.get(seat)
    if not seat_info:
        return None
    return seat_info.get("上下")  # "上段" / "下段"

def scan_class_images(root: Path, cls: str) -> List[Path]:
    cls_dir = root / cls
    if not cls_dir.is_dir():
        return []
    return [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def tag_candidates(paths: List[Path], label: str, store_meta: Dict):
    rows = []
    for p in paths:
        name = p.name
        score = parse_score2(name)
        store, seat = parse_store_and_seat(name)
        updown = get_updown(store_meta, store, seat)
        rows.append({
            "path": p,
            "file": name,
            "source": label,   # "OK" / "WRONG" / "BASE"
            "score": score,
            "store": store,
            "seat": seat,
            "updown": updown,  # "上段" / "下段" / None
        })
    return rows

def filter_ok_wrong(ok_rows, wrong_rows):
    # ※ご要望どおり：OK対象 = 0.9 以下、間違い= 0.5 以下
    ok_sel = [r for r in ok_rows if (r["score"] is not None and r["score"] <= 0.9)]
    wrong_sel = [r for r in wrong_rows if (r["score"] is not None and r["score"] <= 0.5)]
    return ok_sel, wrong_sel

def choose_balanced(cands: List[dict], need: int, want_updown=True) -> List[dict]:
    if need <= 0 or not cands:
        return []
    random.shuffle(cands)

    ups = [c for c in cands if c["updown"] == "上段"]
    downs = [c for c in cands if c["updown"] == "下段"]
    selected = []

    if want_updown:
        half = need // 2
        q_up = min(len(ups), half)
        q_down = min(len(downs), need - q_up)
        # 足りない分は相互補完
        lack = need - (q_up + q_down)
        if lack > 0:
            # 余っている方から補う
            extra_pool = ups if len(ups) - q_up > len(downs) - q_down else downs
            extra_take = min(lack, len(extra_pool) - (q_up if extra_pool is ups else q_down))
            if extra_pool is ups:
                q_up += max(0, extra_take)
            else:
                q_down += max(0, extra_take)
    else:
        q_up = 0
        q_down = 0

    def pick_by_store_balance(pool: List[dict], quota: int, already: List[dict]) -> List[dict]:
        taken = []
        from collections import defaultdict as dd
        store_cnt = Counter([s["store"] for s in already])
        remain = pool.copy()
        while quota > 0 and remain:
            by_store = dd(list)
            for c in remain:
                by_store[c["store"]].append(c)
            # 採用回数が少ない店舗から1枚ずつ取る
            order = sorted(by_store.keys(), key=lambda k: store_cnt[k])
            new_remain = []
            for st in order:
                if quota <= 0:
                    # quota満了、残り戻す
                    for v in by_store.values():
                        new_remain.extend(v)
                    break
                cand_list = by_store[st]
                c = cand_list.pop()
                taken.append(c)
                store_cnt[st] += 1
                quota -= 1
                if cand_list:
                    new_remain.extend(cand_list)
            remain = new_remain
        return taken

    if want_updown:
        selected.extend(pick_by_store_balance(ups, q_up, selected))
        selected.extend(pick_by_store_balance(downs, q_down, selected))

    if len(selected) < need:
        used = set(id(x) for x in selected)
        rest_pool = [c for c in cands if id(c) not in used]
        selected.extend(pick_by_store_balance(rest_pool, need - len(selected), selected))

    return selected[:need]

def build_for_class(cls: str,
                    ok_root: Path, wrong_root: Path, base_root: Path,
                    store_meta: Dict,
                    per_split: Dict[str, int]) -> Dict[str, List[dict]]:
    ok_rows   = tag_candidates(scan_class_images(ok_root, cls), "OK", store_meta)
    wrong_rows= tag_candidates(scan_class_images(wrong_root, cls), "WRONG", store_meta)
    base_rows = tag_candidates(scan_class_images(base_root, cls), "BASE", store_meta)

    ok_sel, wrong_sel = filter_ok_wrong(ok_rows, wrong_rows)

    result = {}
    for split, need in per_split.items():
        pool = wrong_sel + ok_sel + base_rows
        chosen = choose_balanced(pool, need, want_updown=True)
        used_ids = set(id(x) for x in chosen)
        wrong_sel = [x for x in wrong_sel if id(x) not in used_ids]
        ok_sel    = [x for x in ok_sel if id(x) not in used_ids]
        base_rows = [x for x in base_rows if id(x) not in used_ids]
        result[split] = chosen
    return result

def write_excel(manifest_rows: List[dict], out_path: Path):
    df = pd.DataFrame(manifest_rows)
    df = df.rename(columns={
        "file": "画像名",
        "store": "店舗コード",
        "updown": "上下",
        "score": "精度",
        "source": "対象",
        "class": "クラス",
        "split": "データ分割",
        "out_path": "出力先",
    })
    cols = ["データ分割", "クラス", "画像名", "店舗コード", "上下", "精度", "対象", "出力先"]
    df = df[cols]
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="manifest", index=False)

def read_classes_from_excel(xlsx: Path, sheet: str, col: str) -> List[str]:
    df = pd.read_excel(xlsx, sheet_name=sheet)
    if col not in df.columns:
        raise SystemExit(f"Excel '{xlsx}' シート '{sheet}' に列 '{col}' が見つかりません。列名を確認してください。")
    classes = [str(x) for x in df[col].dropna().astype(str).tolist()]
    # 空白や重複を整理
    classes = sorted(set(s.strip() for s in classes if s.strip()))
    return classes

def main():
    ap = argparse.ArgumentParser(description="Curate dataset for classes listed in Excel (class name column).")
    ap.add_argument("--ok-root", type=Path, required=True)
    ap.add_argument("--wrong-root", type=Path, required=True)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--store-json", type=Path, required=True, help="店舗とseat→上下の対応JSON")
    ap.add_argument("--excel-classes", type=Path, required=True, help="クラス一覧を含むExcel")
    ap.add_argument("--sheet", type=str, default="summary", help="Excelシート名（既定: summary）")
    ap.add_argument("--col", type=str, default="class name", help="クラス名の列（既定: 'class name'）")
    ap.add_argument("--dest", type=Path, required=True, help="出力先ルート。中に train/val/test/クラス を作成")
    ap.add_argument("--train", type=int, default=140)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--excel", type=Path, default=Path("summary.xlsx"))
    ap.add_argument("--move", action="store_true", help="コピーではなく移動する（既定はコピー）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    random.seed(42)

    # Excelから対象クラスを取得
    target_classes = read_classes_from_excel(args.excel_classes, args.sheet, args.col)

    # 実在チェック（3ルートの和集合と突き合わせ）
    existing_classes = set()
    for root in (args.ok_root, args.wrong_root, args.base_root):
        if root.is_dir():
            existing_classes |= {p.name for p in root.iterdir() if p.is_dir()}
    missing = [c for c in target_classes if c not in existing_classes]
    if missing:
        print(f"[WARN] 以下のクラスはok/wrong/baseのいずれにも見つかりませんでした: {missing}")

    store_meta = load_store_json(args.store_json)
    per_split = {"train": args.train, "val": args.val, "test": args.test}

    manifest = []
    for cls in target_classes:
        plan = build_for_class(cls, args.ok_root, args.wrong_root, args.base_root, store_meta, per_split)
        for split, items in plan.items():
            out_dir = args.dest / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in items:
                dst = out_dir / r["path"].name
                if not args.dry_run:
                    if args.move:
                        shutil.move(str(r["path"]), str(dst))
                    else:
                        shutil.copy2(str(r["path"]), str(dst))
                manifest.append({
                    "split": split,
                    "class": cls,
                    "file": r["file"],
                    "store": r["store"],
                    "updown": r["updown"],
                    "score": r["score"],
                    "source": r["source"],  # OK / WRONG / BASE
                    "out_path": str(dst),
                })

    if manifest:
        write_excel(manifest, args.excel)

    print(f"完了: {len(manifest)} 枚を選定。Excel: {args.excel}")
    if args.dry_run:
        print("※ --dry-run のためファイル操作は未実施。")

if __name__ == "__main__":
    main()
