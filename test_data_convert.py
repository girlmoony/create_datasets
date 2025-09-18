from pathlib import Path
import numpy as np
import cv2

def png_to_raw_variants(
    src_png_path: str,
    out_dir: str,
    resize_to: tuple | None = None,            # 例: (224,224). Noneなら原寸
    mean: tuple = (0.485, 0.456, 0.406),       # 学習時の値に合わせる
    std:  tuple = (0.229, 0.224, 0.225)        # 学習時の値に合わせる
):
    """
    PNGを読み込み、RGB基準で以下を保存:
      1) rgb_hwc_u8.raw     : HxWx3, RGB, uint8, interleaved
      2) rgb_chw_u8.raw     : 3xHxW, RGB, uint8, planar (CHW)
      3) nchw_f32_no_norm.bin: 3xHxW, float32, 0–1スケールのみ (/255)
      4) nchw_f32_norm.bin  : 3xHxW, float32, /255 → (x-mean)/std
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 読み込み（日本語パス対応） ---
    buf = np.fromfile(src_png_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)       # BGR / BGRA / GRAY
    if img is None:
        raise ValueError(f"Failed to read: {src_png_path}")

    # --- チャンネル整理：BGR/BGRA/GRAY → RGB ---
    if img.ndim == 2:                                   # GRAY → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                               # BGRA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # BGR → RGB

    # --- リサイズ（学習と同じ補間に合わせる） ---
    if resize_to is not None:
        img_rgb = cv2.resize(img_rgb, resize_to, interpolation=cv2.INTER_LINEAR)

    H, W, _ = img_rgb.shape

    stem = Path(src_png_path).stem
    # 1) HWC, uint8, interleaved（RGBRGB...）
    (out_dir / f"{stem}_{W}x{H}_rgb_hwc_u8.raw").write_bytes(
        np.ascontiguousarray(img_rgb, dtype=np.uint8).tobytes()
    )

    # 2) CHW, uint8, planar（R面→G面→B面）
    chw_u8 = np.transpose(img_rgb, (2, 0, 1)).copy()    # 3xHxW
    (out_dir / f"{stem}_{W}x{H}_rgb_chw_u8.raw").write_bytes(chw_u8.tobytes())

    # 3) CHW, float32, 0–1スケールのみ（正規化なし）
    x01 = img_rgb.astype(np.float32) / 255.0            # HWC
    chw_f32_no = np.transpose(x01, (2, 0, 1)).copy()    # 3xHxW
    (out_dir / f"{stem}_{W}x{H}_nchw_f32_no_norm.bin").write_bytes(chw_f32_no.tobytes())

    # 4) CHW, float32, /255 → (x-mean)/std（正規化あり：ご指定の書き方）
    x = img_rgb.astype(np.float32) / 255.0              # HWC
    # --- 正規化（in-placeでチャンネルごとに、質問のスタイル） ---
    x[..., 0] = (x[..., 0] - mean[0]) / std[0]          # R
    x[..., 1] = (x[..., 1] - mean[1]) / std[1]          # G
    x[..., 2] = (x[..., 2] - mean[2]) / std[2]          # B
    chw_f32_norm = np.transpose(x, (2, 0, 1)).copy()    # 3xHxW
    (out_dir / f"{stem}_{W}x{H}_nchw_f32_norm.bin").write_bytes(chw_f32_norm.tobytes())

    # 参考：ブロードキャスト一括版（上のin-placeと等価）
    # mean_arr = np.array(mean, dtype=np.float32).reshape(1,1,3)
    # std_arr  = np.array(std,  dtype=np.float32).reshape(1,1,3)
    # x = (x01 - mean_arr) / std_arr
    # chw_f32_norm = np.transpose(x, (2,0,1)).copy()

    meta = {
        "width": W, "height": H, "channels": 3,
        "layouts": {
            "rgb_hwc_u8.raw": "HWC, RGB, uint8, interleaved",
            "rgb_chw_u8.raw": "CHW, RGB, uint8, planar",
            "nchw_f32_no_norm.bin": "CHW, float32, [0,1], no mean/std",
            "nchw_f32_norm.bin": "CHW, float32, normalized (/255→(x-mean)/std)"
        },
        "mean": mean, "std": std
    }
    return meta

# 実行例
if __name__ == "__main__":
    meta = png_to_raw_variants(
        src_png_path="テスト画像.png",
        out_dir="out_raws",
        resize_to=(224, 224),             # 学習時の入力解像度に合わせる
        mean=(0.485, 0.456, 0.406),       # 学習時と同じ
        std=(0.229, 0.224, 0.225)         # 学習時と同じ
    )
    print(meta)
