# annotate_move.py
import argparse, csv, os, shutil, sys
from pathlib import Path

try:
    import cv2
    USE_CV2 = True
except Exception:
    USE_CV2 = False
    from PIL import Image

def show_image(img_path):
    if USE_CV2:
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = img.shape[:2]
        scale = min(1000 / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        cv2.imshow("image (q: quit, Enter: skip, u: undo)", img)
        cv2.waitKey(1)
    else:
        # Pillowでの簡易表示（OS既定ビューア）
        Image.open(img_path).show()

def close_window():
    if USE_CV2:
        cv2.destroyAllWindows()

def read_classes(class_file):
    with open(class_file, encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    # 数字IDでもラベル名でもOK。ここでは入力文字列をそのまま使う。
    return classes

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def move_or_copy(src: Path, dst: Path, mode="move"):
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

def write_log(log_path, rows):
    newfile = not Path(log_path).exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["src", "dst", "class", "mode"])
        w.writerows(rows)

def collect_images(root):
    root = Path(root)
    # 誤認識フォルダ配下のサブフォルダ群を走査して画像を集める
    folders = [p for p in root.iterdir() if p.is_dir()]
    items = []
    for fd in sorted(folders):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            imgs.extend(fd.glob(ext))
        if imgs:
            items.append((fd, sorted(imgs)))
    return items

def prompt(msg):
    try:
        return input(msg)
    except EOFError:
        return "q"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miscls_root", required=True, help="誤認識フォルダのパス")
    ap.add_argument("--out_root", required=True, help="出力データセットのルート (例: dataset/train)")
    ap.add_argument("--classes", required=True, help="クラス一覧ファイル (1行1クラスID or ラベル名)")
    ap.add_argument("--mode", choices=["move", "copy"], default="move", help="移動 or コピー")
    args = ap.parse_args()

    classes = read_classes(args.classes)
    print(f"[INFO] クラス一覧 ({len(classes)}件): {classes}")

    items = collect_images(args.miscls_root)
    if not items:
        print("画像が見つかりませんでした。")
        return

    undo_stack = []
    log_rows = []

    for folder, images in items:
        print(f"\n=== フォルダ: {folder.name} / 画像枚数: {len(images)} ===")
        ans = prompt(f"このフォルダ内の **全画像** を同一クラスにしますか？ (y/n, qで終了) > ")
        if ans.lower() == "q":
            break

        if ans.lower() == "y":
            print("クラス候補:", classes)
            cls = prompt("正しいクラスID/ラベルを入力（例: 107）> ").strip()
            out_dir = Path(args.out_root) / cls
            moved = []
            for img in images:
                dst = out_dir / img.name
                move_or_copy(img, dst, args.mode)
                moved.append((img, dst, cls))
            undo_stack.append(moved)
            log_rows.extend([(str(s), str(d), cls, args.mode) for s, d, cls in moved])
            print(f"→ {len(moved)}枚を {out_dir} へ {args.mode} しました。")
            continue

        # 個別割当
        for img in images:
            print(f"\n--- {img} ---")
            try:
                show_image(img)
            except Exception as e:
                print(f"(画像表示スキップ: {e})")

            print("クラス候補:", classes)
            cmd = prompt("正しいクラスID/ラベルを入力 / Enter=スキップ / u=直前Undo / q=終了 > ").strip()
            if cmd.lower() == "q":
                close_window()
                write_log("annotation_log.csv", log_rows)
                print("終了します。ログを書き出しました。")
                return
            if cmd == "":
                continue
            if cmd.lower() == "u":
                if undo_stack:
                    last = undo_stack.pop()
                    for s, d, cls in last:
                        # 元に戻す（コピーの場合は削除のみ）
                        if args.mode == "copy":
                            if Path(d).exists():
                                Path(d).unlink()
                        else:
                            ensure_dir(Path(s).parent)
                            shutil.move(d, s)
                    print("Undo 完了。")
                else:
                    print("Undo スタックが空です。")
                continue

            # 割当実行
            cls = cmd
            out_dir = Path(args.out_root) / cls
            dst = out_dir / img.name
            move_or_copy(img, dst, args.mode)
            undo_stack.append([(img, dst, cls)])
            log_rows.append((str(img), str(dst), cls, args.mode))
            print(f"→ {dst} へ {args.mode} しました。")

    close_window()
    write_log("annotation_log.csv", log_rows)
    print("完了。annotation_log.csv に記録しました。")

if __name__ == "__main__":
    main()
