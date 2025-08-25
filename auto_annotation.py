# preprocess.py
from PIL import Image, ImageOps, ImageCms
import os, json, hashlib, imagehash, cv2
from pathlib import Path

SRC = Path("raw_images")
DST = Path("sushi-256")
CLASSES = [d.name for d in SRC.iterdir() if d.is_dir()]

def to_srgb(img):
    try:
        icc = img.info.get('icc_profile', None)
        if icc:
            srgb = ImageCms.createProfile("sRGB")
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            img = ImageCms.profileToProfile(img, src, srgb, outputMode='RGB')
    except Exception:
        pass
    return img.convert("RGB")

def square_256(img):
    img = ImageOps.exif_transpose(img)
    img = to_srgb(img)
    # 等比缩放到最长边=256
    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
    # 反射填充到方形
    w, h = img.size
    if w == h == 256: return img
    pad = ImageOps.expand(img, border=(0,0,256-w,256-h), fill=None)  # 先右下填，下一步再镜像拼
    bg = Image.new("RGB", (256,256))
    bg.paste(img, ((256-w)//2, (256-h)//2))
    return bg

def blur_score(pil_img):
    import numpy as np
    arr = np.array(pil_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

duplicates = set()
hash_map = {}

for c in CLASSES:
    out_dir = DST / "all" / c
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (SRC/c).glob("*"):
        try:
            img = Image.open(p)
            ph = imagehash.phash(img)
            if ph in hash_map:  # 近重复
                continue
            img = square_256(img)
            if blur_score(img) < 80:  # 自行调参
                continue
            fname = hashlib.md5(str(p).encode()).hexdigest() + ".png"
            img.save(out_dir/fname, format="PNG", compress_level=3, optimize=True)
            hash_map[ph] = True
        except Exception:
            continue
