#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import csv
from PIL import Image
from torch.utils.data import Dataset

def _read_rows(manifest_csv: Path):
    with manifest_csv.open(newline='', encoding='utf-8') as f:
        rdr = csv.DictReader((ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")))
        for r in rdr:
            yield {
                "path": r["path"].strip(),
                "label": r["label"].strip(),
                "split": r["split"].strip().lower()
            }

def build_class_to_idx(manifest_csv: Path) -> Dict[str, int]:
    labels = set()
    for r in _read_rows(manifest_csv):
        labels.add(r["label"])
    return {c: i for i, c in enumerate(sorted(labels))}

class ManifestDataset(Dataset):
    """
    合成済みマニフェスト（path,label,split）を読み、指定splitの (full_path, label_idx) を保持。
    - path は相対パス推奨。full_path = data_root / path で解決。
    - class_to_idx は train/val/test で共通のものを渡すと ID が安定。
    """
    def __init__(self,
                 manifest_csv: Path,
                 data_root: Path,
                 split: str = "train",
                 class_to_idx: Optional[Dict[str, int]] = None,
                 type: str = "train",          # ★ 追加: "train" / "val" / "test"
                 image_size: int = 224,
                 strict_exists: bool = True):
        self.manifest_csv = Path(manifest_csv)
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.type = (type or self.split).lower()   # 明示未指定なら split に合わせる
        self.image_size = image_size

        if class_to_idx is None:
            class_to_idx = build_class_to_idx(self.manifest_csv)
        self.class_to_idx = class_to_idx

        items: List[Tuple[Path, int]] = []
        missing = 0
        for r in _read_rows(self.manifest_csv):
            if r["split"] != self.split:
                continue
            full = (self.data_root / r["path"]).resolve()
            if not full.exists():
                if strict_exists:
                    missing += 1
                    continue
            items.append((full, self.class_to_idx[r["label"]]))

        self.items = items
        if strict_exists and missing > 0:
            print(f"[ManifestDataset] Missing files ignored: {missing}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # ★ __getitem__ 内で手動定義（初回のみ生成→キャッシュ）
        if self._tfms is None:
            if self.type == "train":
                self._tfms = T.Compose([
                    T.Resize((self.image_size, self.image_size)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ])
            else:  # val / test など
                self._tfms = T.Compose([
                    T.Resize((self.image_size, self.image_size)),
                    T.ToTensor(),
                ])

        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        img = self._tfms(img)
        return img, y


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, random, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from dataset_manifest import ManifestDataset, build_class_to_idx

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True, help="画像のルート（相対pathはここ基準）")
    ap.add_argument("--manifest", type=Path, required=True, help="合成済みCSV（compose_manifest.pyの出力）")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("runs/exp1"))
    return ap.parse_args()

def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    # 変換
    train_tfms = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor()])
    val_tfms   = T.Compose([T.Resize((224,224)), T.ToTensor()])

    # クラス辞書（安定）
    class_to_idx = build_class_to_idx(args.manifest)

    # Dataset & Loader
    train_ds = ManifestDataset(args.manifest, args.data_root, split="train",
                               class_to_idx=class_to_idx, transform=train_tfms)
    val_ds   = ManifestDataset(args.manifest, args.data_root, split="val",
                               class_to_idx=class_to_idx, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # モデル
    num_classes = len(class_to_idx)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        train_loss = loss_sum / max(total,1)
        train_acc = correct / max(total,1)

        # ---- val ----
        model.eval()
        v_total, v_correct, v_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss_sum += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += x.size(0)
        val_loss = v_loss_sum / max(v_total,1)
        val_acc = v_correct / max(v_total,1)

        print(f"[{epoch:02d}] train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")

        # save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "class_to_idx": class_to_idx}, args.out / "best.pt")

    print(f"best val acc: {best_val_acc:.3f}")

if __name__ == "__main__":
    main()


# 2) 学習（v2を使用）
python train.py \
  --data_root /mnt/ds_project \
  --manifest manifests/v2.csv \
  --epochs 10 --batch_size 64 --workers 8 \
  --out runs/v2_resnet18

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18
import torch.nn as nn
from dataset_manifest import ManifestDataset

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)  # runs/exp1/best.pt
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    return ap.parse_args()

def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}

    tfms = T.Compose([T.Resize((224,224)), T.ToTensor()])
    ds = ManifestDataset(args.manifest, args.data_root, split=args.split,
                         class_to_idx=class_to_idx, transform=tfms)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    # モデル構築
    num_classes = len(class_to_idx)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    print(f"{args.split} acc: {correct / max(total,1):.3f} ({correct}/{total})")

if __name__ == "__main__":
    main()
# 3) 評価（test split を評価する例。v2.csv 内に test を用意した前提）
python eval.py \
  --data_root /mnt/ds_project \
  --manifest manifests/v2.csv \
  --checkpoint runs/v2_resnet18/best.pt \
  --split test



既存コードの修正ポイント（チェックリスト）
データセット生成
旧：ImageFolder(root=...) など“ディレクトリ走査で自動ラベリング”
新：MnifestDataset(manifest, data_root, split=...) に置き換え
class_to_idx は train から作って val/test に渡す（上のtrain.py参照）
データ分割（train/val/test）
旧：random_split や “フォルダ分け”で分割
新：マニフェスト側の split 列で固定（コード側では split= 指定のみ）
クラス数の取り方
旧：len(dataset.classes)
新：len(class_to_idx)（train.py 参照）
入力パス
旧：--data_dir 1本
新：--data_root（画像の物理ルート）＋ --manifest（合成済CSV）
ログ/再現性
合成に使った --base と --delta のセット、生成した manifest のパスを実験ログに記録
seed 固定（train.py の set_seed）
分散学習（必要なら）
DistributedSampler(train_ds) を DataLoader に付与するだけでOK（上記は単GPU最小実装）
