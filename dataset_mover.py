#!/usr/bin/env python3
# save as dataset_mover.py
# usage examples:
#   1) 任意フォルダを移動:
#      python dataset_mover.py move-folders --csv folders.csv --dest /dst/root --on-exist rename --dry-run
#   2) クラスを分割ごとに移動:
#      python dataset_mover.py move-classes --src /src/classes_root \
#          --splits-csv /path/to/train.csv /path/to/val.csv /path/to/test.csv \
#          --dest /dst/dataset_root --on-exist skip

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Optional

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unique_rename_path(dst: Path) -> Path:
    """dst が存在するとき dst_1, dst_2 ... として重複しないパスを返す"""
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def move_dir(src: Path, dst: Path, on_exist: str = "skip", dry_run: bool = False) -> str:
    """
    src ディレクトリを dst へ移動。
    on_exist: 'skip' | 'overwrite' | 'rename'
    """
    if not src.exists():
        return f"[SKIP] not found: {src}"

    if dst.exists():
        if on_exist == "skip":
            return f"[SKIP] already exists: {dst}"
        elif on_exist == "overwrite":
            # shutil.move は既存ディレクトリへ「中身を入れ込む」動きになる。
            # 確実に置き換えたいなら事前削除。
            if not dry_run:
                shutil.rmtree(dst)
        elif on_exist == "rename":
            dst = unique_rename_path(dst)
        else:
            return f"[ERROR] invalid on_exist: {on_exist}"

    ensure_dir(dst.parent)
    if dry_run:
        return f"[DRYRUN] move {src} -> {dst}"
    shutil.move(str(src), str(dst))
    return f"[OK] move {src} -> {dst}"

# -------------------------
# サブコマンド1: move-folders
# -------------------------
def cmd_move_folders(csv_path: Path, dest_root: Path, on_exist: str, dry_run: bool) -> None:
    """
    CSVで指定したフォルダ群を dest_root 直下へ移動（直下にフォルダ名で並べる）。
    CSVフォーマット（ヘッダあり/なしどちらでも可）:
      - 単一列: フォルダの絶対/相対パス
      - または列名 'path' を含む
    """
    dest_root = dest_root.resolve()
    ensure_dir(dest_root)

    # CSV読み
    rows: List[str] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sniff = csv.Sniffer()
        sample = f.read(4096)
        f.seek(0)
        dialect = sniff.sniff(sample) if sample else csv.excel
        reader = csv.reader(f, dialect)
        # ヘッダ判定
        f.seek(0)
        has_header = sniff.has_header(sample) if sample else False
        if has_header:
            dict_reader = csv.DictReader(f, dialect=dialect)
            for r in dict_reader:
                path = r.get("path") or next((v for v in r.values() if v), "")
                if path:
                    rows.append(path.strip())
        else:
            for cols in reader:
                if not cols:
                    continue
                rows.append(cols[0].strip())

    logs = []
    for src_str in rows:
        src = Path(src_str).resolve()
        dst = dest_root / src.name
        logs.append(move_dir(src, dst, on_exist=on_exist, dry_run=dry_run))

    print("\n".join(logs))

# -------------------------
# サブコマンド2: move-classes
# -------------------------
def read_list_from_csv(csv_file: Path, column: Optional[str] = None) -> List[str]:
    """
    クラス名のリストをCSVから読む。
    - ヘッダあり: 指定列名、なければ最初の列
    - ヘッダなし: 1列目
    空行はスキップ。
    """
    items: List[str] = []
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        data = f.read(4096)
        f.seek(0)
        sniff = csv.Sniffer()
        dialect = sniff.sniff(data) if data else csv.excel
        f.seek(0)
        has_header = sniff.has_header(data) if data else False

        if has_header:
            reader = csv.DictReader(f, dialect=dialect)
            # 列名指定がなければ先頭列
            if column is None:
                if reader.fieldnames:
                    column = reader.fieldnames[0]
            if column is None:
                raise ValueError("CSVに列がありません")
            for r in reader:
                val = (r.get(column) or "").strip()
                if val:
                    items.append(val)
        else:
            reader = csv.reader(f, dialect)
            for row in reader:
                if row and row[0].strip():
                    items.append(row[0].strip())
    return items

def cmd_move_classes(src_root: Path, splits_csv: List[Path], dest_root: Path,
                     colname: Optional[str], on_exist: str, dry_run: bool) -> None:
    """
    src_root 直下にクラス名フォルダがある想定。
      src_root/
        ├─ classA/
        ├─ classB/
        └─ classC/
    splits_csv は train/val/test の3つ（順序は任意、ファイル名に 'train','val','test' が含まれていれば自動割当）。
    各CSVにはクラス名（= フォルダ名）を1列で列挙（ヘッダ有無は不問）。
    """
    # 出力先に train/val/test を用意
    split_map = {"train": None, "val": None, "test": None}
    for c in splits_csv:
        name = c.name.lower()
        if "train" in name and split_map["train"] is None:
            split_map["train"] = c
        elif "val" in name and split_map["val"] is None:
            split_map["val"] = c
        elif "test" in name and split_map["test"] is None:
            split_map["test"] = c

    missing = [k for k, v in split_map.items() if v is None]
    if missing:
        raise SystemExit(f"train/val/test それぞれのCSVが必要です。不足: {missing}")

    # 目的のパスに train/val/test を作成
    for s in ("train", "val", "test"):
        ensure_dir(dest_root.joinpath(s))

    logs = []
    for split in ("train", "val", "test"):
        csv_file = split_map[split]
        classes = read_list_from_csv(csv_file, column=colname)
        for cls in classes:
            src_dir = src_root / cls
            dst_dir = dest_root / split / cls
            logs.append(move_dir(src_dir, dst_dir, on_exist=on_exist, dry_run=dry_run))

    print("\n".join(logs))

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Move folders or class-folders into target train/val/test structure."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # move-folders
    ap1 = sub.add_parser("move-folders", help="CSVで列挙したフォルダをdest直下へ移動")
    ap1.add_argument("--csv", type=Path, required=True, help="移動対象フォルダ一覧CSV（1列: path）")
    ap1.add_argument("--dest", type=Path, required=True, help="移動先のルートフォルダ（直下に並べる）")
    ap1.add_argument("--on-exist", choices=["skip", "overwrite", "rename"], default="skip",
                     help="移動先に同名フォルダがある場合の挙動")
    ap1.add_argument("--dry-run", action="store_true", help="実際には移動せずログのみ")

    # move-classes
    ap2 = sub.add_parser("move-classes", help="train/val/testのCSVに列挙されたクラスを所定の構造へ移動")
    ap2.add_argument("--src", dest="src_root", type=Path, required=True,
                     help="クラスフォルダが並んでいる元ルート")
    ap2.add_argument("--splits-csv", type=Path, nargs=3, required=True,
                     help="train/val/test それぞれのCSV（ファイル名に train/val/test を含めてください）")
    ap2.add_argument("--dest", dest="dest_root", type=Path, required=True,
                     help="出力ルート。中に train/ val/ test/ を作成し、その直下へクラスを移動")
    ap2.add_argument("--col", dest="colname", type=str, default=None,
                     help="CSVにヘッダがある場合の列名（指定がなければ先頭列）")
    ap2.add_argument("--on-exist", choices=["skip", "overwrite", "rename"], default="skip",
                     help="移動先に同名フォルダがある場合の挙動")
    ap2.add_argument("--dry-run", action="store_true", help="実際には移動せずログのみ")

    args = ap.parse_args()

    if args.cmd == "move-folders":
        cmd_move_folders(args.csv, args.dest, args.on_exist, args.dry_run)
    elif args.cmd == "move-classes":
        cmd_move_classes(args.src_root, args.splits_csv, args.dest_root,
                         args.colname, args.on_exist, args.dry_run)
    else:
        raise SystemExit("unknown command")

if __name__ == "__main__":
    main()
