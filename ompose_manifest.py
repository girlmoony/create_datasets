#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベース vb.csv と 1..N 個の delta.csv を合成し、有効マニフェスト (path,label,split) を出力。
相対パスは train/eval 時に --data_root で解決します。
"""
import csv
import argparse
from collections import OrderedDict
from pathlib import Path

def load_base(base_csv: Path):
    m = OrderedDict()  # path -> (label, split)
    with base_csv.open(newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            p = r["path"].strip()
            m[p] = (r["label"].strip(), r["split"].strip())
    return m

def apply_delta(base_map: OrderedDict, delta_csv: Path):
    with delta_csv.open(newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            op = r["op"].strip().upper()
            p  = r["path"].strip()
            if op == "ADD":
                base_map[p] = (r["label"].strip(), r["split"].strip())
            elif op == "DROP":
                base_map.pop(p, None)
            elif op == "MOVE":
                lab, _ = base_map.get(p, (r["label"].strip(), None))
                base_map[p] = (lab, r["split"].strip())
            elif op == "RELABEL":
                _, sp = base_map.get(p, (None, r["split"].strip()))
                base_map[p] = (r["label"].strip(), sp)
            else:
                raise ValueError(f"Unknown op: {op}")
    return base_map

def dump_manifest(m: OrderedDict, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["path","label","split"])
        for p, (lab, sp) in m.items():
            w.writerow([p, lab, sp])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="vb.csv")
    ap.add_argument("--delta", type=Path, action="append", default=[], help="*.delta.csv (複数可、上から順に適用)")
    ap.add_argument("--out", type=Path, required=True, help="合成結果のCSV")
    args = ap.parse_args()

    m = load_base(args.base)
    for d in args.delta:
        m = apply_delta(m, d)
    dump_manifest(m, args.out)
    print(f"Wrote: {args.out} (rows={len(m)})")

if __name__ == "__main__":
    main()


# 1) 合成（例：v2 を作る）
python compose_manifest.py \
  --base manifests/vb.csv \
  --delta manifests/v2.delta.csv \
  --out manifests/v2.csv

sample
ベース（例）：manifests/vb.csv
path,label,split
images/Vb/dog/001.jpg,dog,train
images/Vb/dog/002.jpg,dog,train
images/Vb/cat/101.jpg,cat,val
images/Vb/cat/222.jpg,cat,train


差分（例）：manifests/v2.delta.csv

操作は ADD / DROP / MOVE / RELABEL
op,path,label,split
ADD,additions/bird/007.jpg,bird,train
DROP,images/Vb/cat/101.jpg,,
MOVE,images/Vb/cat/222.jpg,cat,val

使い方例：

v2: vb.csv + v2.delta.csv
v3: vb.csv + v3.delta.csv
v4: vb.csv + v3.delta.csv + v2.delta.csv（v3にv2の追加を重ねる）

