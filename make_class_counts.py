# save as make_class_counts.py
import os
from pathlib import Path
from typing import Iterable
import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def count_images_in_dir(d: Path) -> int:
    if not d.exists() or not d.is_dir():
        return 0
    # 再帰で画像拡張子をカウント
    return sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

def make_class_counts(paths: Iterable[str], excel_path: str,
                      summary_sheet: str = "summary",
                      class_col: str = "class name") -> pd.DataFrame:
    paths = [str(p) for p in paths]
    # クラス一覧をExcelから読む
    df_summary = pd.read_excel(excel_path, sheet_name=summary_sheet)
    classes = [str(x) for x in df_summary[class_col].dropna().astype(str).tolist()]

    # 結果テーブルの骨組み
    out_rows = []
    for cls in classes:
        row = {"class name": cls}
        total = 0
        for i, base in enumerate(paths, start=1):
            base_path = Path(base)
            cls_dir = base_path / cls   # ルート直下にクラス名フォルダがある想定
            cnt = count_images_in_dir(cls_dir)
            row[f"path{i}"] = str(cls_dir)
            row[f"count{i}"] = cnt
            total += cnt
        row["total"] = total
        out_rows.append(row)

    # 列順：class name → total → path1/count1 → path2/count2 → ...
    cols = ["class name", "total"]
    for i in range(1, len(paths)+1):
        cols += [f"path{i}", f"count{i}"]
    df_out = pd.DataFrame(out_rows)[cols]
    return df_out

if __name__ == "__main__":
    # 使い方例
    paths = [
        "/data/dataset_v1",     # ルート1（直下にクラス名フォルダ）
        "/data/dataset_extra",  # ルート2
    ]
    excel_path = "classes.xlsx"  # summaryシートにclass name列があるExcel
    df = make_class_counts(paths, excel_path, summary_sheet="summary", class_col="class name")
    out_xlsx = "class_counts.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="class_counts", index=False)
    print(f"書き出し: {out_xlsx}")
