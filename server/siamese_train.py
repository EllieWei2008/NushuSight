# -*- coding: utf-8 -*-
"""
siamese_train.py — 女书孪生网络训练脚本
架构: VGG16 主干 + L1 距离头 + BCE 损失
运行环境: Google Colab (GPU T4/A100) 或本地 CUDA

用法:
  # Colab — 挂载 Drive 后直接运行
  python siamese_train.py --data_root /content/drive/MyDrive/nushu/data

  # 本地
  python siamese_train.py --data_root ./data
"""

import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ── 超参数 ───────────────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 64        # 对数量（每 batch 64 对）
LR          = 1e-4
STEP_SIZE   = 10        # StepLR 每 N epoch 降一次 LR
GAMMA       = 0.5
SEED        = 42
IMG_SIZE    = 224       # VGG16 输入尺寸
SAVE_EVERY  = 5         # 每 N epoch 保存一次 checkpoint

# ── 固定随机种子 ──────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 数据集 ────────────────────────────────────────────────────────────────────
class NushuPairDataset(Dataset):
    """
    从 PaddleOCR 格式的 label 文件中读取图像，
    每次 __getitem__ 返回一对图像及标签（1=同类，0=异类）。
    """

    def __init__(self, label_file: str, data_root: str, transform=None, seed: int = SEED):
        self.data_root = Path(data_root)
        self.transform = transform
        self.rng = random.Random(seed)

        # 读取标注文件：path\tclass_index
        self.samples: list[tuple[str, int]] = []
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                self.samples.append((parts[0], int(parts[1])))

        # 按类别分组，方便采样同类/异类
        self.class_to_imgs: dict[int, list[str]] = {}
        for path, cls in self.samples:
            self.class_to_imgs.setdefault(cls, []).append(path)
        self.class_list = list(self.class_to_imgs.keys())

    def __len__(self) -> int:
        # 返回 2× 样本数（一半正对，一半负对）
        return len(self.samples) * 2

    def _load(self, rel_path: str) -> Image.Image:
        full = self.data_root / rel_path
        img = Image.open(full).convert("L")      # 灰度
        img = img.convert("RGB")                  # VGG16 需要 3 通道
        return img

    def __getitem__(self, idx: int):
        # 偶数 idx → 正样本对；奇数 idx → 负样本对
        is_positive = (idx % 2 == 0)
        base_path, base_cls = self.samples[idx // 2]
        img1 = self._load(base_path)

        if is_positive:
            candidates = [p for p in self.class_to_imgs[base_cls] if p != base_path]
            if not candidates:
                candidates = self.class_to_imgs[base_cls]
            img2_path = self.rng.choice(candidates)
            label = 1.0
        else:
            neg_cls = self.rng.choice([c for c in self.class_list if c != base_cls])
            img2_path = self.rng.choice(self.class_to_imgs[neg_cls])
            label = 0.0

        img2 = self._load(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ── 模型 ──────────────────────────────────────────────────────────────────────
class SiameseNet(nn.Module):
    """
    VGG16 特征提取器（共享权重）+ L1 距离 → 相似度得分
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained
                           else None)
        # 取 features + avgpool，输出 512 维向量
        self.features = vgg.features
        self.avgpool  = vgg.avgpool
        self.flatten  = nn.Flatten()
        self.proj     = nn.Linear(512 * 7 * 7, 512)
        self.relu     = nn.ReLU(inplace=True)
        self.head     = nn.Linear(512, 1)
        self.sigmoid  = nn.Sigmoid()

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.relu(self.proj(x))
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.forward_one(x1)
        f2 = self.forward_one(x2)
        diff = torch.abs(f1 - f2)      # L1 距离
        return self.sigmoid(self.head(diff)).squeeze(1)


# ── 训练 / 验证循环 ────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for img1, img2, labels in loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            preds = model(img1, img2)
            loss  = criterion(preds, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            pred_cls    = (preds >= 0.5).float()
            correct    += (pred_cls == labels).sum().item()
            total      += len(labels)

    return total_loss / total, correct / total


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default=os.environ.get("DATA_ROOT", "./data"))
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--save_dir",   default="./checkpoints")
    parser.add_argument("--no_pretrain",action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root  = Path(args.data_root)
    save_dir   = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 数据变换
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.1),  # 女书字符方向敏感，小概率翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = NushuPairDataset(data_root / "train_labels.txt", data_root / "dataset", train_tf)
    val_ds   = NushuPairDataset(data_root / "val_labels.txt",   data_root / "dataset", val_tf,   seed=99)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"Train pairs: {len(train_ds):,}  Val pairs: {len(val_ds):,}")

    model     = SiameseNet(pretrained=not args.no_pretrain).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {vl_loss:.4f} acc {vl_acc:.4f}")

        # 保存最佳模型
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_dir / "siamese_best.pth")
            print(f"  → Best model saved (val_acc={best_val_acc:.4f})")

        # 定期 checkpoint（防 Colab 断线）
        if epoch % SAVE_EVERY == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            }
            torch.save(ckpt, save_dir / f"ckpt_ep{epoch:02d}.pth")

    print(f"\nTraining done. Best val accuracy: {best_val_acc:.4f}")

    # 保存训练历史
    import json
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {save_dir}/history.json")


if __name__ == "__main__":
    main()
