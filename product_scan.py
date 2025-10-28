#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

import pandas as pd

SUBSETS = ["train", "val", "test"]

@dataclass
class FolderHit:
    base: str
    subset: str
    path: str

@dataclass
class CountSummary:
    total_png: int = 0
    by_type: Dict[str, int] = None
    by_updown: Dict[str, int] = None
    by_lr: Dict[str, int] = None
    def __post_init__(self):
        from collections import defaultdict as dd
        if self.by_type is None: self.by_type = dd(int)
        if self.by_updown is None: self.by_updown = dd(int)
        if self.by_lr is None: self.by_lr = dd(int)
    def add(self, t: Optional[str], updown: Optional[str], lr: Optional[str]):
        self.total_png += 1
        self.by_type[t or "不明"] += 1
        self.by_updown[updown or "不明"] += 1
        self.by_lr[lr or "不明"] += 1

def load_store_info(json_path: Path) -> Dict[str, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_product_folders(product: str, bases: List[Path]) -> List[FolderHit]:
    hits: List[FolderHit] = []
    for base in bases:
        for subset in SUBSETS:
            root = base / subset
            if not root.exists(): continue
            try:
                for d in root.iterdir():
                    if d.is_dir() and d.name.startswith(f"{product}_"):
                        hits.append(FolderHit(str(base), subset, str(d)))
            except FileNotFoundError:
                continue
    return hits

def parse_png_metadata(png_path: Path, storeinfo: Dict[str, dict], pattern: re.Pattern) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = pattern.match(png_path.name)
    if not m: return (None, None, None)
    store = m.group("store")
    cam = m.group("cam")
    info = storeinfo.get(store)
    if not info: return (None, None, None)
    model_type = info.get("型") or info.get("type")
    cam_map = info.get(cam) or {}
    updown = cam_map.get("上下") or cam_map.get("updown")
    lr = cam_map.get("左右") or cam_map.get("leftright")
    return (model_type, updown, lr)

def scan_pngs(folder_paths: Iterable[str], storeinfo: Dict[str, dict], regex: str) -> CountSummary:
    pat = re.compile(regex)
    summary = CountSummary()
    seen: Set[Path] = set()
    for p in folder_paths:
        folder = Path(p)
        if not folder.exists(): continue
        for png in folder.rglob("*.png"):
            try:
                real = png.resolve()
            except Exception:
                real = png
            if real in seen: continue
            seen.add(real)
            t, up, lr = parse_png_metadata(png, storeinfo, pat)
            summary.add(t, up, lr)
    return summary

def forward_fill(series: pd.Series) -> pd.Series:
    return series.ffill()

def main():
    parser = argparse.ArgumentParser(description="Scan products in Excel and analyze dataset folders.")
    parser.add_argument("--excel", required=True)
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--product_col", default="A")
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--bases", nargs="+", required=True)
    parser.add_argument("--storeinfo", required=True)
    parser.add_argument("--out_sheet", default="out")
    parser.add_argument("--filename_regex", default=r'.*?(?P<store>\d{3,4}).*?(?P<cam>\d{2}).*?\.png$')
    args = parser.parse_args()

    if len(args.bases) > 10:
        raise ValueError("最多10個までのベースパスを指定してください (--bases)")

    excel_path = Path(args.excel)
    bases = [Path(p) for p in args.bases]
    storeinfo_path = Path(args.storeinfo)
    if not excel_path.exists(): raise FileNotFoundError(excel_path)
    if not storeinfo_path.exists(): raise FileNotFoundError(storeinfo_path)

    usecols = args.product_col if args.label_col is None else f"{args.product_col},{args.label_col}"
    df = pd.read_excel(excel_path, sheet_name=args.sheet, usecols=usecols, header=0)
    cols = list(df.columns)
    if len(cols) == 1:
        df.columns = ["product"]; df["label"] = ""
    else:
        df.columns = ["product", "label"]

    df["product"] = forward_fill(df["product"])
    df["product"] = df["product"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    df = df[df["product"] != ""].reset_index(drop=True)

    storeinfo = load_store_info(storeinfo_path)

    records_rows: List[Dict[str, str]] = []
    details_rows: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        product = str(row["product"]).strip()
        label = str(row.get("label", "") or "").strip()

        hits = find_product_folders(product, bases)
        found_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for h in hits:
            found_map[(h.base, h.subset)].append(h.path)

        summary = scan_pngs([h.path for h in hits], storeinfo, args.filename_regex)

        summary_row = {
            "product": product,
            "label": label,
            "total_png": summary.total_png,
            "型_新型": summary.by_type.get("新型", 0),
            "型_旧型": summary.by_type.get("旧型", 0),
            "型_不明": summary.by_type.get("不明", 0),
            "上下_上段": summary.by_updown.get("上段", 0),
            "上下_下段": summary.by_updown.get("下段", 0),
            "上下_不明": summary.by_updown.get("不明", 0),
            "左右_左": summary.by_lr.get("左", 0),
            "左右_右": summary.by_lr.get("右", 0),
            "左右_不明": summary.by_lr.get("不明", 0),
        }
        for base in bases:
            for sb in SUBSETS:
                key = (str(base), sb)
                paths = found_map.get(key, [])
                summary_row[f"found[{base.name}][{sb}]"] = ";".join(paths)

        records_rows.append(summary_row)

        for (base, subset), paths in found_map.items():
            for p in paths:
                details_rows.append({
                    "product": product,
                    "label": label,
                    "base": base,
                    "subset": subset,
                    "folder": p,
                })

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        pd.DataFrame(records_rows).to_excel(writer, sheet_name=args.out_sheet, index=False)
        if details_rows:
            pd.DataFrame(details_rows).to_excel(writer, sheet_name=f"{args.out_sheet}_details", index=False)

    print(f"Wrote {len(records_rows)} products to '{args.out_sheet}' in {excel_path}")

if __name__ == "__main__":
    main()

python product_scan.py \
  --excel /path/to/workbook.xlsx \
  --sheet Sheet1 \
  --product_col A \
  --label_col B \
  --bases /mnt/datasetsA /mnt/datasetsB \
  --storeinfo /path/to/storeInfo.json \
  --out_sheet out
