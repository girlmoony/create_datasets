# pip install torch torchvision opencv-python
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

# ===== 共通設定 =====
SIZE = 256
MEAN = [0.485, 0.456, 0.406]   # RGB
STD  = [0.229, 0.224, 0.225]   # RGB
NUM_CLASSES = 286

# --- Dataset: BGRで読み → RGB変換 → PIL.Image で返す ---
class SushiDataset(Dataset):
    def __init__(self, samples, transform):
        """
        samples: List[(img_path, class_idx)]
        transform: torchvision.transforms.Compose
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)  # torchvision は PIL/RGB が扱いやすい
        x = self.transform(img)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# ===== パターン1：ベースライン（安全・軽量） =====
def get_train_tf_pattern1():
    return T.Compose([
        # 256合わせ（元が256なら実質NOP）
        T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(SIZE),
        T.RandomHorizontalFlip(p=0.5),
        # 色相は控えめ
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BICUBIC),

        T.ToTensor(),                       # [0,1] Tensor
        T.Normalize(mean=MEAN, std=STD),    # RGB基準
    ])

# ===== パターン2：照明・テカりロバスト =====
def get_train_tf_pattern2():
    return T.Compose([
        T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomResizedCrop(size=SIZE, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),

        # Photometricをやや強め
        T.ColorJitter(brightness=0.35, contrast=0.30, saturation=0.20, hue=0.02),
        # torchvision にはCLAHEはないため、近似として以下を併用：
        T.RandomAutocontrast(p=0.3),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        # 反射ムラ/質感の揺れをならす（軽いシャープ調整）
        T.RandomAdjustSharpness(sharpness_factor=1.2, p=0.3),

        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

# ===== パターン3：背景・皿ばらつきロバスト（構図強化） =====
def get_train_tf_pattern3():
    return T.Compose([
        T.RandomResizedCrop(size=SIZE, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomPerspective(distortion_scale=0.2, p=0.2, interpolation=T.InterpolationMode.BICUBIC),

        # 彩度/色相は控えめ（ネタの色相を壊さない）
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.20, hue=0.02),

        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        # 部分隠れノイズへの耐性（CoarseDropout代替）
        T.RandomErasing(p=0.25, scale=(0.01, 0.05), ratio=(0.3, 3.3), value='random'),
    ])

# ===== パターン4：混合学習（MixUp/CutMix） =====
# Transform自体はパターン2 or 3を使い、バッチ単位でMixUp/CutMixを適用
def one_hot(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix

def cutmix_batch(x, y, alpha=0.8):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    N, C, H, W = x.size()
    idx = torch.randperm(N, device=x.device)
    cx, cy = np.random.randint(W), np.random.randint(H)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))
    x1 = max(cx - w // 2, 0); x2 = min(cx + w // 2, W)
    y1 = max(cy - h // 2, 0); y2 = min(cy + h // 2, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_mix = lam_adj * y + (1 - lam_adj) * y[idx]
    return x, y_mix

def collate_fn_mixup(batch, alpha=0.2, num_classes=NUM_CLASSES):
    xs, ys = zip(*batch)
    x = torch.stack(xs, 0)
    y = one_hot(torch.tensor(ys), num_classes=num_classes)
    return mixup_batch(x, y, alpha)

def collate_fn_cutmix(batch, alpha=0.8, num_classes=NUM_CLASSES):
    xs, ys = zip(*batch)
    x = torch.stack(xs, 0)
    y = one_hot(torch.tensor(ys), num_classes=num_classes)
    return cutmix_batch(x, y, alpha)

# ===== 検証/テスト（決定的）参考 =====
def get_eval_tf():
    return T.Compose([
        T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(SIZE),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
