import argparse
import shutil
from pathlib import Path
from typing import List, Set, Union, Dict, Any
import pandas as pd
import logging
import sys
from datetime import datetime

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def setup_logger(log_path: Path, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("sync_classes")
    logger.setLevel(logging.DEBUG)
    # 既存ハンドラ重複を避ける
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def read_class_names(excel_path: Union[str, Path], sheet_name: Union[int, str] = 0,
                     column: Union[int, str] = 0, logger: logging.Logger = None) -> List[str]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    series = df.iloc[:, column] if isinstance(column, int) else df[column]
    seen = set()
    classes: List[str] = []
    for v in series.dropna().astype(str).str.strip():
        if v and v not in seen:
            classes.append(v)
            seen.add(v)
    if logger:
        logger.info(f"Excel から {len(classes)} 件のクラス名を読み込みました")
    return classes

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def count_images_in_dir(dir_path: Path) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.rglob("*") if is_image_file(p))

def copy_dir(src: Path, dst: Path, overwrite: bool,
             logger: logging.Logger, dry_run: bool = False) -> None:
    if not src.exists():
        logger.error(f"コピー元が見つかりません: {src}")
        return
    if overwrite and dst.exists():
        logger.info(f"  - 既存を削除: {dst}")
        if not dry_run:
            shutil.rmtree(dst)
    if dst.exists():
        logger.info(f"  - 既に存在のためスキップ: {dst}")
        return
    logger.info(f"  - コピー: {src} -> {dst}")
    if not dry_run:
        shutil.copytree(src, dst)

def ensure_dir_exists(p: Path, logger: logging.Logger) -> None:
    if not p.exists():
        logger.error(f"ディレクトリが見つかりません: {p}")
        raise FileNotFoundError(f"ディレクトリが見つかりません: {p}")
    if not p.is_dir():
        logger.error(f"ディレクトリではありません: {p}")
        raise NotADirectoryError(f"ディレクトリではありません: {p}")

def process(excel_path: Path, dir_a: Path, dir_b: Path,
            sheet_name: Union[int, str] = 0, column: Union[int, str] = 0,
            threshold: int = 5, dry_run: bool = False,
            results_csv: Union[Path, None] = None,
            results_xlsx: Union[Path, None] = None,
            logger: logging.Logger = None) -> None:

    logger.info("=== 設定 ===")
    logger.info(f"Excel        : {excel_path}")
    logger.info(f"Sheet        : {sheet_name}")
    logger.info(f"Column       : {column}")
    logger.info(f"A フォルダ   : {dir_a}")
    logger.info(f"B フォルダ   : {dir_b}")
    logger.info(f"閾値(枚数)   : {threshold}")
    logger.info(f"Dry-run      : {dry_run}")
    logger.info("==============")

    ensure_dir_exists(dir_a, logger)
    ensure_dir_exists(dir_b, logger)

    classes = read_class_names(excel_path, sheet_name=sheet_name, column=column, logger=logger)
    if not classes:
        logger.warning("Excel からクラス名が読み込めませんでした。（先頭列/指定列が空かも）")
        return

    # 結果を貯める（後で DataFrame 化）
    rows: List[Dict[str, Any]] = []

    # ステップ1: A に無いものを B から補完
    logger.info("\n=== ステップ1: A に存在しないクラスを補完 ===")
    for cls in classes:
        a_cls = dir_a / cls
        b_cls = dir_b / cls
        exists_in_a_before = a_cls.exists()
        exists_in_b = b_cls.exists()
        images_before = count_images_in_dir(a_cls) if exists_in_a_before else 0

        action = "skipped_ok"
        note = ""

        if not exists_in_a_before:
            if exists_in_b:
                logger.info(f"[追加] {cls}")
                copy_dir(b_cls, a_cls, overwrite=False, dry_run=dry_run, logger=logger)
                action = "copied_new_from_b"
            else:
                logger.warning(f"  ! B にも存在しません: {b_cls}")
                action = "missing_in_b"
                note = "AにもBにも存在せず未作成"

        images_after = images_before
        if action == "copied_new_from_b" and not dry_run:
            images_after = count_images_in_dir(a_cls)

        rows.append({
            "class": cls,
            "exists_in_a_before": exists_in_a_before,
            "exists_in_b": exists_in_b,
            "images_in_a_before": images_before,
            "action_step1": action,
            "images_in_a_after_step1": images_after,
            "note_step1": note,
        })

    # ステップ2: 枚数が閾値以下なら B から上書き
    logger.info("\n=== ステップ2: 画像枚数チェック & 必要なら上書き ===")
    for cls in classes:
        a_cls = dir_a / cls
        b_cls = dir_b / cls

        exists_in_a = a_cls.exists()
        exists_in_b = b_cls.exists()
        images_before = count_images_in_dir(a_cls) if exists_in_a else 0

        action = "none"
        note = ""
        images_after = images_before

        if not exists_in_a:
            logger.warning(f"[欠落] {cls} は A に未作成")
            action = "missing_in_a"
            note = "Step1でも作成されず"
        else:
            logger.info(f"[枚数] {cls}: {images_before} 枚")
            if images_before <= threshold:
                if exists_in_b:
                    logger.info(f"  -> {cls}: {threshold} 枚以下のため B から上書きコピー")
                    copy_dir(b_cls, a_cls, overwrite=True, dry_run=dry_run, logger=logger)
                    action = "overwritten_from_b"
                    if not dry_run:
                        images_after = count_images_in_dir(a_cls)
                else:
                    logger.warning(f"  ! B に {cls} が無く、上書きできません")
                    action = "cannot_overwrite_missing_in_b"
            else:
                action = "kept_as_is"

        rows.append({
            "class": cls,
            "exists_in_a_before": exists_in_a,
            "exists_in_b": exists_in_b,
            "images_in_a_before": images_before,
            "action_step2": action,
            "images_in_a_after_step2": images_after,
            "note_step2": note,
        })

    # DataFrame へ
    df = pd.DataFrame(rows)

    # 既定ファイル名（未指定時）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if results_csv is None:
        results_csv = Path(f"sync_results_{ts}.csv")
    if results_xlsx is None:
        results_xlsx = Path(f"sync_results_{ts}.xlsx")

    # 保存
    try:
        df.to_csv(results_csv, index=False, encoding="utf-8-sig")
        logger.info(f"結果CSVを書き出しました: {results_csv.resolve()}")
    except Exception as e:
        logger.error(f"CSV書き出し失敗: {e}")

    try:
        with pd.ExcelWriter(results_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="results")
        logger.info(f"結果Excelを書き出しました: {results_xlsx.resolve()}")
    except Exception as e:
        logger.error(f"Excel書き出し失敗: {e}")

    logger.info("\n処理完了。")

def main():
    parser = argparse.ArgumentParser(
        description="Excel のクラス名リストに基づいて A フォルダを B から補完/上書きし、結果をログと表で保存"
    )
    parser.add_argument("--excel_path", type=Path, required=True, help="Excel ファイルパス（.xlsx）")
    parser.add_argument("--sheet_name", default=0, help="シート名またはインデックス（既定: 0）")
    parser.add_argument("--column", default=0, help="列名またはインデックス（既定: 0 = 先頭列）")
    parser.add_argument("--dir_a", type=Path, required=True, help="A フォルダのパス")
    parser.add_argument("--dir_b", type=Path, required=True, help="B フォルダのパス")
    parser.add_argument("--threshold", type=int, default=5, help="この枚数以下なら B で上書き（既定: 5）")
    parser.add_argument("--dry-run", action="store_true", help="実際にはコピーせず動作だけ確認")
    parser.add_argument("--log_path", type=Path, default=Path("sync_classes.log"),
                        help="ログファイルパス（既定: sync_classes.log）")
    parser.add_argument("--results_csv", type=Path, default=None, help="結果CSVの出力先")
    parser.add_argument("--results_xlsx", type=Path, default=None, help="結果Excelの出力先")

    args = parser.parse_args()

    def try_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return x

    sheet = try_int(args.sheet_name)
    col = try_int(args.column)
    logger = setup_logger(args.log_path, verbose=True)

    process(
        excel_path=args.excel_path,
        dir_a=args.dir_a,
        dir_b=args.dir_b,
        sheet_name=sheet,
        column=col,
        threshold=args.threshold,
        dry_run=args.dry_run,
        results_csv=args.results_csv,
        results_xlsx=args.results_xlsx,
        logger=logger,
    )

if __name__ == "__main__":
    main()
