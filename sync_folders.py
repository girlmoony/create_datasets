###
python sync_classes.py \
  --excel_path "classes.xlsx" \
  --sheet_name 0 \
  --column 0 \
  --dir_a "path/to/A" \
  --dir_b "path/to/B" \
  --threshold 5
###



import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Set, Union

import pandas as pd


IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def read_class_names(
    excel_path: Union[str, Path],
    sheet_name: Union[int, str] = 0,
    column: Union[int, str] = 0,
) -> List[str]:
    """
    Excel の指定列からクラス名（フォルダ名）を読み込む。
    先頭行から順に非空セルのみを抽出し、重複は除去するが順序は維持する。
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    # 列の取り出し（インデックス or ラベル両対応）
    if isinstance(column, int):
        series = df.iloc[:, column]
    else:
        series = df[column]

    seen = set()
    classes: List[str] = []
    for v in series.dropna().astype(str).str.strip():
        if v and v not in seen:
            classes.append(v)
            seen.add(v)
    return classes


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def count_images_in_dir(dir_path: Path) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.rglob("*") if is_image_file(p))


def copy_dir(src: Path, dst: Path, overwrite: bool, dry_run: bool = False) -> None:
    """
    ディレクトリをコピー。overwrite=True の場合は dst を先に削除してから丸ごとコピー。
    """
    if not src.exists():
        raise FileNotFoundError(f"コピー元が見つかりません: {src}")

    if overwrite and dst.exists():
        print(f"  - 既存を削除: {dst}")
        if not dry_run:
            shutil.rmtree(dst)

    if dst.exists():
        # overwrite=False で既に存在する場合は、コピーしない（何もしない）
        print(f"  - 既に存在のためスキップ: {dst}")
        return

    print(f"  - コピー: {src} -> {dst}")
    if not dry_run:
        shutil.copytree(src, dst)


def ensure_dir_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"ディレクトリが見つかりません: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"ディレクトリではありません: {p}")


def process(
    excel_path: Path,
    dir_a: Path,
    dir_b: Path,
    sheet_name: Union[int, str] = 0,
    column: Union[int, str] = 0,
    threshold: int = 5,
    dry_run: bool = False,
) -> None:
    print("=== 設定 ===")
    print(f"Excel        : {excel_path}")
    print(f"Sheet        : {sheet_name}")
    print(f"Column       : {column}")
    print(f"A フォルダ   : {dir_a}")
    print(f"B フォルダ   : {dir_b}")
    print(f"閾値(枚数)   : {threshold}")
    print(f"Dry-run      : {dry_run}")
    print("==============")

    ensure_dir_exists(dir_a)
    ensure_dir_exists(dir_b)

    classes = read_class_names(excel_path, sheet_name=sheet_name, column=column)
    if not classes:
        print("Excel からクラス名が読み込めませんでした。（先頭列/指定列が空かも）")
        return

    print(f"クラス数: {len(classes)}")
    missing_in_b: List[str] = []
    newly_copied: List[str] = []
    overwritten: List[str] = []
    warn_missing_after: List[str] = []

    # 1) A に存在しないクラスを B からコピー
    print("\n=== ステップ1: A に存在しないクラスを補完 ===")
    for cls in classes:
        a_cls = dir_a / cls
        b_cls = dir_b / cls
        if not a_cls.exists():
            print(f"[追加] {cls}")
            if b_cls.exists():
                copy_dir(b_cls, a_cls, overwrite=False, dry_run=dry_run)
                newly_copied.append(cls)
            else:
                print(f"  ! B にも存在しません: {b_cls}")
                missing_in_b.append(cls)
        else:
            print(f"[OK ] {cls} は A に存在")

    # 2) A 直下の全クラスの画像数を確認し、閾値以下なら B から上書き
    print("\n=== ステップ2: 画像枚数チェック & 必要なら上書き ===")
    for cls in classes:
        a_cls = dir_a / cls
        b_cls = dir_b / cls
        if not a_cls.exists():
            print(f"[欠落] {cls} は A に未作成（ステップ1で B 側も無かった可能性）")
            warn_missing_after.append(cls)
            continue

        count_a = count_images_in_dir(a_cls)
        print(f"[枚数] {cls}: {count_a} 枚")
        if count_a <= threshold:
            if b_cls.exists():
                print(f"  -> {cls}: 枚数が {threshold} 枚以下のため B から上書きコピー")
                copy_dir(b_cls, a_cls, overwrite=True, dry_run=dry_run)
                overwritten.append(cls)
            else:
                print(f"  ! B に {cls} が無く、上書きできません")
                missing_in_b.append(cls)

    # 3) 結果サマリ
    print("\n=== 結果 ===")
    if newly_copied:
        print(f"新規コピー（A に無かったもの）: {len(newly_copied)} 件")
        for c in newly_copied:
            print(f"  - {c}")
    else:
        print("新規コピー: なし")

    if overwritten:
        print(f"上書きコピー（閾値以下）      : {len(overwritten)} 件")
        for c in overwritten:
            print(f"  - {c}")
    else:
        print("上書きコピー: なし")

    if missing_in_b:
        print(f"B 側に存在しなかったクラス     : {len(missing_in_b)} 件")
        for c in sorted(set(missing_in_b)):
            print(f"  - {c}")
    else:
        print("B 側欠落: なし")

    if warn_missing_after:
        print(f"最終的に A に存在しないクラス  : {len(warn_missing_after)} 件")
        for c in warn_missing_after:
            print(f"  - {c}")
    else:
        print("A 側は Excel 記載のクラスを満たしています（上記 B 欠落除く）。")

    print("\n処理完了。")


def main():
    parser = argparse.ArgumentParser(
        description="Excel のクラス名リストに基づいて A フォルダを B から補完/上書きするツール"
    )
    parser.add_argument("--excel_path", type=Path, required=True, help="Excel ファイルパス（.xlsx）")
    parser.add_argument("--sheet_name", default=0, help="シート名またはインデックス（既定: 0）")
    parser.add_argument("--column", default=0, help="列名またはインデックス（既定: 0 = 先頭列）")
    parser.add_argument("--dir_a", type=Path, required=True, help="A フォルダのパス")
    parser.add_argument("--dir_b", type=Path, required=True, help="B フォルダのパス")
    parser.add_argument("--threshold", type=int, default=5, help="この枚数以下なら B で上書き（既定: 5）")
    parser.add_argument("--dry-run", action="store_true", help="実際にはコピーせず動作だけ確認")

    args = parser.parse_args()

    # sheet_name/column を int に解釈できるなら int にしておく
    def try_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return x

    sheet = try_int(args.sheet_name)
    col = try_int(args.column)

    process(
        excel_path=args.excel_path,
        dir_a=args.dir_a,
        dir_b=args.dir_b,
        sheet_name=sheet,
        column=col,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
