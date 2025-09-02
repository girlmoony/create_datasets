import argparse
import shutil
from pathlib import Path
from typing import List, Union
import pandas as pd
import logging
import sys

def setup_logger(log_path: Path | None = None, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("copy_a_to_b")
    logger.setLevel(logging.DEBUG)
    # 重複防止
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    if log_path:
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


def read_class_names(
    excel_path: Union[str, Path],
    sheet_name: Union[int, str] = 0,
    column: Union[int, str] = 0,
) -> List[str]:
    """Excel の指定列からクラス(フォルダ)名を読み込む。空白は除去、重複は最初の出現を採用。"""
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    series = df.iloc[:, column] if isinstance(column, int) else df[column]
    seen = set()
    classes: List[str] = []
    for v in series.dropna().astype(str).str.strip():
        if v and v not in seen:
            classes.append(v)
            seen.add(v)
    return classes


def copy_class_dir(src_root: Path, dst_root: Path, cls: str, overwrite: bool, dry_run: bool, logger: logging.Logger):
    src = src_root / cls
    dst = dst_root / cls

    if not src.exists() or not src.is_dir():
        logger.warning(f"[スキップ] A に存在しません: {src}")
        return

    if dst.exists():
        if overwrite:
            logger.info(f"[上書き] {dst} を削除してコピーし直します")
            if not dry_run:
                shutil.rmtree(dst)
        else:
            logger.info(f"[スキップ] 既に存在: {dst}（--overwrite 未指定）")
            return

    logger.info(f"[コピー] {src} -> {dst}")
    if not dry_run:
        # Python 3.8+ なら dirs_exist_ok=True でもOKだが、上で削除しているため不要
        shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Excel のフォルダ名リストに基づき、A から B へ該当フォルダをコピーします。"
    )
    parser.add_argument("--excel_path", type=Path, required=True, help="Excel ファイルパス（.xlsx）")
    parser.add_argument("--sheet_name", default=0, help="シート名またはインデックス（既定: 0）")
    parser.add_argument("--column", default=0, help="列名またはインデックス（既定: 0 = 先頭列）")
    parser.add_argument("--dir_a", type=Path, required=True, help="コピー元 A フォルダのパス")
    parser.add_argument("--dir_b", type=Path, required=True, help="コピー先 B フォルダのパス")
    parser.add_argument("--overwrite", action="store_true", help="B に同名フォルダがある場合は上書き")
    parser.add_argument("--dry-run", action="store_true", help="実際にはコピーせず動作のみ表示")
    parser.add_argument("--log_path", type=Path, default=None, help="ログ出力先（省略時はコンソールのみ）")

    args = parser.parse_args()

    # 数値可能なら int に
    def try_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return x

    sheet = try_int(args.sheet_name)
    col = try_int(args.column)

    logger = setup_logger(args.log_path, verbose=True)

    # 前提チェック
    if not args.dir_a.exists() or not args.dir_a.is_dir():
        logger.error(f"A フォルダが見つからないかディレクトリではありません: {args.dir_a}")
        sys.exit(1)

    args.dir_b.mkdir(parents=True, exist_ok=True)

    # リスト読み込み
    class_list = read_class_names(args.excel_path, sheet_name=sheet, column=col)
    if not class_list:
        logger.error("Excel からフォルダ名が読み込めませんでした。列やシート指定を確認してください。")
        sys.exit(1)

    logger.info("=== 設定 ===")
    logger.info(f"Excel      : {args.excel_path}")
    logger.info(f"Sheet      : {sheet}")
    logger.info(f"Column     : {col}")
    logger.info(f"A (src)    : {args.dir_a}")
    logger.info(f"B (dst)    : {args.dir_b}")
    logger.info(f"Overwrite  : {args.overwrite}")
    logger.info(f"Dry-run    : {args.dry_run}")
    logger.info(f"対象クラス数: {len(class_list)}")
    logger.info("==============")

    # コピー実行
    for cls in class_list:
        copy_class_dir(args.dir_a, args.dir_b, cls, overwrite=args.overwrite, dry_run=args.dry_run, logger=logger)

    logger.info("処理完了。")

if __name__ == "__main__":
    main()
