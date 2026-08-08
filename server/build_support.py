# -*- coding: utf-8 -*-
"""
build_support.py — 离线预计算 400 个字符的 support 特征向量

支持三种模式：
  merged : corrections 手写图 > Phase2 训练集 > 合成图（推荐，三层叠加）
  dataset: 用 data/phase2/train/ 的增强图（每类多张平均）
  base   : 用 data/base/ 的字体渲染图（每类1张，快速）

用法:
  python demo/build_support.py --mode merged    # 推荐，每次纠错后运行
  python demo/build_support.py --mode dataset
  python demo/build_support.py --mode base
"""

import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "phase1"))
from siamese_train import SiameseNet, IMG_SIZE


class SimpleImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L").convert("RGB")
        return self.transform(img), idx


def build(model_path: str, base_dir: str, char_dict: str,
          dataset_dir: str, out_dir: str, mode: str, n_per_class: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: {mode}")

    model = SiameseNet(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {model_path}")

    with open(char_dict, encoding="utf-8") as f:
        chars = [line.rstrip("\n") for line in f if line.strip()]
    print(f"Character dict: {len(chars)} chars")

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats  = []
    index  = []

    for i, char in enumerate(chars):
        cp  = ord(char)
        uid = f"U{cp:05X}"

        if mode == "base":
            # 单张字体图
            png = Path(base_dir) / f"{uid}.png"
            if not png.exists():
                feats.append(torch.zeros(512))
                index.append({"idx": i, "unicode_id": uid, "char": char,
                               "codepoint": hex(cp), "available": False, "n_samples": 0})
                continue
            img    = Image.open(png).convert("L").convert("RGB")
            tensor = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.forward_one(tensor).squeeze(0).cpu()
            n = 1

        else:
            # 训练集多张图取平均
            char_dir = Path(dataset_dir) / "train" / uid
            if not char_dir.exists():
                feats.append(torch.zeros(512))
                index.append({"idx": i, "unicode_id": uid, "char": char,
                               "codepoint": hex(cp), "available": False, "n_samples": 0})
                continue

            pngs = sorted(char_dir.glob("*.png"))[:n_per_class]
            if not pngs:
                feats.append(torch.zeros(512))
                index.append({"idx": i, "unicode_id": uid, "char": char,
                               "codepoint": hex(cp), "available": False, "n_samples": 0})
                continue

            ds     = SimpleImageDataset(pngs, tf)
            loader = DataLoader(ds, batch_size=32, num_workers=0)
            batch_feats = []
            with torch.no_grad():
                for imgs, _ in loader:
                    batch_feats.append(model.forward_one(imgs.to(device)).cpu())
            feat = torch.cat(batch_feats, dim=0).mean(dim=0)   # 多张取平均
            n    = len(pngs)

        feats.append(feat)
        index.append({"idx": i, "unicode_id": uid, "char": char,
                       "codepoint": hex(cp), "available": True, "n_samples": n})

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(chars)}] {uid} done (n={n})")

    support = torch.stack(feats)
    torch.save(support, out_dir / "support_features.pt")

    with open(out_dir / "support_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total_samples = sum(e["n_samples"] for e in index)
    print(f"\nSupport features: {support.shape} → {out_dir}/support_features.pt")
    print(f"Mode: {mode}, total samples used: {total_samples}")
    print(f"Index saved → {out_dir}/support_index.json")


def build_merged(model_path: str, base_dir: str, char_dict: str,
                 dataset_dir: str, corrections_dir: str,
                 out_dir: str, n_per_class: int = 20):
    """
    三层叠加支撑集（推荐方式）：
      优先级1: corrections/crops 手写纠错图（人工验证，最精准）
      优先级2: Phase2 训练集图均值（多书写者，泛化好）
      优先级3: 合成 base 图（兜底）
    """
    from collections import defaultdict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Mode: merged")

    # 加载模型
    model = SiameseNet(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def feat(p: Path) -> torch.Tensor:
        img = Image.open(p).convert("L").convert("RGB")
        t   = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.forward_one(t).squeeze(0).cpu()

    with open(char_dict, encoding="utf-8") as f:
        chars = [l.rstrip() for l in f if l.strip()]

    # 层1：收集 corrections/crops 手写图
    corr_feats: dict[str, list] = defaultdict(list)
    corr_path = Path(corrections_dir)
    if corr_path.exists():
        for session in sorted(corr_path.iterdir()):
            if not session.is_dir():
                continue
            for png in sorted(session.glob("*.png")):
                uid = png.stem.split("_corr_")[0]
                if uid.startswith("U1B") and len(uid) == 6:
                    corr_feats[uid].append(feat(png))

    # 层2：Phase2 训练集图
    ph2_feats: dict[str, list] = defaultdict(list)
    ph2_train = Path(dataset_dir) / "train"
    if ph2_train.exists():
        for c in chars:
            uid  = f"U{ord(c):05X}"
            pngs = sorted((ph2_train / uid).glob("*_hw*.png"))[:n_per_class]
            if not pngs:
                pngs = sorted((ph2_train / uid).glob("*.png"))[:n_per_class]
            for p in pngs:
                ph2_feats[uid].append(feat(p))

    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = []; index = []
    c_count = p_count = s_count = 0

    for i, c in enumerate(chars):
        uid = f"U{ord(c):05X}"
        if uid in corr_feats and corr_feats[uid]:
            f_vec  = torch.stack(corr_feats[uid]).mean(0)
            source = "corrections"
            c_count += 1
        elif uid in ph2_feats and ph2_feats[uid]:
            f_vec  = torch.stack(ph2_feats[uid]).mean(0)
            source = "phase2_train"
            p_count += 1
        else:
            base = Path(base_dir) / f"{uid}.png"
            f_vec  = feat(base) if base.exists() else torch.zeros(512)
            source = "synthetic"
            s_count += 1
        feats.append(f_vec)
        index.append({"unicode_id": uid, "char": c, "codepoint": hex(ord(c)),
                      "available": True, "source": source, "n_samples": 1})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(chars)}] done")

    support = torch.stack(feats)
    torch.save(support, out_dir / "support_features.pt")
    with open(out_dir / "support_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n支撑集重建完成（merged 模式）:")
    print(f"  corrections 手写图: {c_count} 个字符（优先）")
    print(f"  Phase2 训练集:      {p_count} 个字符（次选）")
    print(f"  合成 base 图:       {s_count} 个字符（兜底）")
    print(f"  → {out_dir}/support_features.pt  {support.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    default="./checkpoints/oneshot/oneshot_best.pth")
    parser.add_argument("--base_dir",      default="./data/base")
    parser.add_argument("--char_dict",     default="./data/char_dict.txt")
    parser.add_argument("--dataset_dir",   default="./data/phase2")
    parser.add_argument("--corrections_dir", default="./demo/corrections/crops")
    parser.add_argument("--out_dir",       default="./demo")
    parser.add_argument("--mode",          default="merged",
                        choices=["merged", "base", "dataset"],
                        help="merged=三层叠加(推荐), dataset=训练集均值, base=单张合成图")
    parser.add_argument("--n_per_class",   type=int, default=20,
                        help="dataset/merged 模式每类取多少张训练集图")
    args = parser.parse_args()

    if args.mode == "merged":
        # 三层叠加：corrections > phase2_train > base
        build_merged(args.model_path, args.base_dir, args.char_dict,
                     args.dataset_dir, args.corrections_dir,
                     args.out_dir, args.n_per_class)
    else:
        build(args.model_path, args.base_dir, args.char_dict,
              args.dataset_dir, args.out_dir, args.mode, args.n_per_class)

