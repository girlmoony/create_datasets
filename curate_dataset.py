#!/usr/bin/env python3
# save as curate_dataset.py
# usage:
#   python curate_dataset.py \
#     --ok-root /data/ok_root \
#     --wrong-root /data/wrong_root \
#     --base-root /data/base_root \
#     --store-json /data/stores_and_cameras.json \
#     --dest /data/curated_out \
#     --train 140 --val 30 --test 30 \
#     --excel summary.xlsx
#
#   # 実際にコピーせず計画だけ見たい場合
#   # --dry-run を付ける
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
#   - 先頭 [] は無視して、2つ目 [] を score2 として取得
#   - 店舗コード (連続数字)
#   - seat は "-08-" のようにハイフンに挟まれた2桁
RE_SCORE2 = re.compile(r"^\[[^\]]*\]\[(?P<score>\d+(?:\.\d+)?)\]")
RE_STORE_SEAT = re.compile(r"^(?:\[[^\]]+\])+\d*(?P<rest>.*)")  # 先頭の[]群はスキップだけ
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
    # 店舗
    m1 = RE_STORE.search(name)
    store = m1.group("store") if m1 else None
    # seat
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
    return seat_info.get("上下")  # "上段" / "下段" 期待

def scan_class_images(root: Path, cls: str) -> List[Path]:
    cls_dir = root / cls
    if not cls_dir.is_dir():
        return []
    paths = []
    for p in cls_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths

def tag_candidates(paths: List[Path], label: str, store_meta: Dict):
    """
    label: "OK" / "WRONG" / "BASE"
    返り値: 辞書のリスト
    """
    rows = []
    for p in paths:
        name = p.name
        score = parse_score2(name)
        store, seat = parse_store_and_seat(name)
        updown = get_updown(store_meta, store, seat)  # None もあり
        rows.append({
            "path": p,
            "file": name,
            "source": label,
            "score": score,
            "store": store,
            "seat": seat,
            "updown": updown,  # "上段" / "下段" / None
        })
    return rows

def filter_ok_wrong(ok_rows, wrong_rows):
    """
    仕様:
      - OKデータ: 2つ目の[] = 推論スコア <= 0.9 を対象
      - 間違いデータ: 精度 <= 0.5 を対象
    """
    ok_sel = [r for r in ok_rows if (r["score"] is not None and r["score"] <= 0.9)]
    wrong_sel = [r for r in wrong_rows if (r["score"] is not None and r["score"] <= 0.5)]
    return ok_sel, wrong_sel

def choose_balanced(cands: List[dict], need: int, want_updown=True) -> List[dict]:
    """
    バランス方針（貪欲）:
      1) まず上下を半々目標に quota を設定（不足側は余った quota を他方に回す）
      2) 各 quota 枠の中で、店舗の採用回数が少ないものを優先して round-robin
      3) None(上下不明) は最後に不足分を補う
    """
    if need <= 0 or not cands:
        return []

    random.shuffle(cands)  # 偏り防止

    # 上下で分割
    ups = [c for c in cands if c["updown"] == "上段"]
    downs = [c for c in cands if c["updown"] == "下段"]
    unknowns = [c for c in cands if c["updown"] not in ("上段", "下段")]

    selected = []

    if want_updown:
        half = need // 2
        q_up = half
        q_down = need - half
        # 足りなければ相互補完
        if len(ups) < q_up:
            q_down += (q_up - len(ups))
            q_up = len(ups)
        if len(downs) < q_down:
            q_up += (q_down - len(downs))
            q_down = len(downs)
    else:
        q_up = 0
        q_down = 0

    def pick_by_store_balance(pool: List[dict], quota: int) -> List[dict]:
        taken = []
        store_cnt = Counter()
        # 先に既選抜の店舗数も考慮
        for s in selected:
            store_cnt[s["store"]] += 1
        # ループで最小カウントの店舗を優先
        remain = pool.copy()
        while quota > 0 and remain:
            # storeごとに代表ひとつずつ回す（過剰な同一店舗偏りを防ぐ）
            by_store = defaultdict(list)
            for c in remain:
                by_store[c["store"]].append(c)
            # 店舗を現在の採用数が少ない順に並べる
            order = sorted(by_store.keys(), key=lambda k: store_cnt[k])
            new_remain = []
            for st in order:
                if quota <= 0:
                    # quota満了、残りを new_remain に積み直して終了
                    new_remain.extend(sum(by_store.values(), []))
                    break
                cand_list = by_store[st]
                c = cand_list.pop()  # ひとつ取る
                taken.append(c)
                store_cnt[st] += 1
                quota -= 1
                # まだ残りがあれば new_remain に戻す
                if cand_list:
                    new_remain.extend(cand_list)
            remain = new_remain
        return taken

    # 上段→下段を先に
    if want_updown:
        selected.extend(pick_by_store_balance(ups, q_up))
        selected.extend(pick_by_store_balance(downs, q_down))

    # 足りない分は unknown と未使用の残りから補完
    if len(selected) < need:
        used = set(id(x) for x in selected)
        rest_pool = [c for c in cands if id(c) not in used]
        take_rest = need - len(selected)
        # 店舗バランス優先で残りを取る
        selected.extend(pick_by_store_balance(rest_pool, take_rest))

    return selected[:need]

def build_for_class(cls: str,
                    ok_root: Path, wrong_root: Path, base_root: Path,
                    store_meta: Dict,
                    per_split: Dict[str, int]) -> Dict[str, List[dict]]:
    """
    各splitで、WRONG優先 → OK(<=0.9) → BASE の順に補充しながら
    店舗＆上下バランスを図る。
    """
    ok_rows   = tag_candidates(scan_class_images(ok_root, cls), "OK", store_meta)
    wrong_rows= tag_candidates(scan_class_images(wrong_root, cls), "WRONG", store_meta)
    base_rows = tag_candidates(scan_class_images(base_root, cls), "BASE", store_meta)

    ok_sel, wrong_sel = filter_ok_wrong(ok_rows, wrong_rows)

    result = {}
    for split, need in per_split.items():
        # ここでは split 別の元データ分割は前提しない（全プールから選ぶ）
        pool = wrong_sel + ok_sel + base_rows
        chosen = choose_balanced(pool, need, want_updown=True)
        # 使ったものは他 split から重複選出しないよう除外
        used_ids = set(id(x) for x in chosen)
        wrong_sel = [x for x in wrong_sel if id(x) not in used_ids]
        ok_sel    = [x for x in ok_sel if id(x) not in used_ids]
        base_rows = [x for x in base_rows if id(x) not in used_ids]
        result[split] = chosen
    return result

def write_excel(manifest_rows: List[dict], out_path: Path):
    df = pd.DataFrame(manifest_rows)
    # 欲しい列順
    cols = ["split", "class", "画像名", "店舗コード", "上下", "精度", "対象", "出力先"]
    # 列名整形
    df = df.rename(columns={
        "file": "画像名",
        "store": "店舗コード",
        "updown": "上下",
        "score": "精度",
        "source": "対象",
        "class": "class",
        "split": "split",
        "out_path": "出力先",
    })
    df = df[cols]
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="manifest", index=False)

def main():
    ap = argparse.ArgumentParser(description="Curate 14-class dataset with store & up/down balancing from OK/WRONG/BASE roots.")
    ap.add_argument("--ok-root", type=Path, required=True)
    ap.add_argument("--wrong-root", type=Path, required=True)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--store-json", type=Path, required=True, help="店舗とseat→上下の対応JSON")
    ap.add_argument("--dest", type=Path, required=True, help="出力先ルート。中に train/val/test/classX を作成")
    ap.add_argument("--train", type=int, default=140)
    ap.add_argument("--val", type=int, default=30)
    ap.add_argument("--test", type=int, default=30)
    ap.add_argument("--excel", type=Path, default=Path("summary.xlsx"))
    ap.add_argument("--move", action="store_true", help="コピーではなく移動する（既定はコピー）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    random.seed(42)

    store_meta = load_store_json(args.store_json)
    per_split = {"train": args.train, "val": args.val, "test": args.test}

    # クラス一覧は3ルートの和集合から推定（14クラス前提だが自動化）
    classes = set()
    for root in (args.ok_root, args.wrong_root, args.base_root):
        if root.is_dir():
            for p in root.iterdir():
                if p.is_dir():
                    classes.add(p.name)
    classes = sorted(list(classes))

    manifest = []
    for cls in classes:
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
                    "source": r["source"],  # 対象: OK / WRONG / BASE
                    "out_path": str(dst),
                })

    if manifest:
        write_excel(manifest, args.excel)

    print(f"完了: {len(manifest)} 枚を選定。Excel: {args.excel}")
    if args.dry_run:
        print("※ --dry-run のためファイル操作は実施していません。")

if __name__ == "__main__":
    main()
