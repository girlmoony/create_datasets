# dataloaders.py
import json
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tfms = T.Compose([
    T.Resize(256, interpolation=InterpolationMode.BICUBIC),
    T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(0.1, 0.1, 0.1, 0.05),
    T.RandomApply([T.GaussianBlur(3)], p=0.1),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tfms = T.Compose([
    T.Resize(256, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class JsonListDataset(Dataset):
    def __init__(self, split_json, split_name, transform=None):
        data = json.load(open(split_json, 'r'))
        self.samples = []
        self.classes = sorted(list(data['train'].keys()))  # 假定三者同类集
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        for c, paths in data[split_name].items():
            for p in paths:
                self.samples.append((p, self.class_to_idx[c]))
        self.transform = transform

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

def make_loaders(root="sushi-256/metadata/splits.json", bs=128, nw=8):
    train_ds = JsonListDataset(root, "train", train_tfms)
    val_ds = JsonListDataset(root, "val", eval_tfms)
    test_ds = JsonListDataset(root, "test", eval_tfms)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader, test_loader
