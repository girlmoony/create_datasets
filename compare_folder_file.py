import os
import hashlib

def file_hash(path, algo="sha256", chunk_size=8192):
    """指定ファイルのハッシュ値を返す"""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def compare_folders(folder1, folder2):
    # フォルダ内のファイル名一覧を取得
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # 両方に存在するファイル名を抽出
    common = files1 & files2

    results = []
    for fname in sorted(common):
        path1 = os.path.join(folder1, fname)
        path2 = os.path.join(folder2, fname)

        # ディレクトリは無視
        if not os.path.isfile(path1) or not os.path.isfile(path2):
            continue

        # ハッシュを比較
        h1 = file_hash(path1)
        h2 = file_hash(path2)

        if h1 == h2:
            results.append((fname, "同一"))
        else:
            results.append((fname, "内容が異なる"))

    return results

# 使い方例
folder_a = "/path/to/folderA"
folder_b = "/path/to/folderB"

for fname, status in compare_folders(folder_a, folder_b):
    print(f"{fname}: {status}")
