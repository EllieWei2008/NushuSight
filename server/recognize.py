# -*- coding: utf-8 -*-
"""
recognize.py — 单字符识别模块
输入: PIL Image（单个字符，任意尺寸）
输出: Top-5 候选列表 [{rank, unicode_id, char, codepoint, distance, confidence}]

依赖 build_support.py 预先生成的 support_features.pt 和 support_index.json
可选: 通过 load_handwriting_labels() 追加手写支撑集，提升手写体识别率
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "phase1"))
from siamese_train import SiameseNet, IMG_SIZE


_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class NushuRecognizer:
    """
    女书字符识别器。
    初始化一次后可反复调用 recognize()。
    支持随时追加手写支撑集，无需重启。
    """

    def __init__(self, model_path: str, demo_dir: str = None,
                 device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # 默认使用脚本自身所在目录，无论从哪里调用
        self._demo_dir = Path(demo_dir) if demo_dir else Path(__file__).parent

        # 加载模型
        self.model = SiameseNet(pretrained=False).to(self.device)
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        if "model" in state:
            state = state["model"]
        self.model.load_state_dict(state)
        self.model.eval()

        # 加载合成支撑集
        feat_path = self._demo_dir / "support_features.pt"
        idx_path  = self._demo_dir / "support_index.json"
        if not feat_path.exists() or not idx_path.exists():
            raise FileNotFoundError(
                f"Support files not found in {self._demo_dir}. "
                "Run: python server/build_support.py"
            )
        self._syn_support = torch.load(feat_path, map_location="cpu",
                                       weights_only=False)
        with open(idx_path, encoding="utf-8") as f:
            self._syn_index = json.load(f)

        # 手写支撑集（初始为空，调用 load_handwriting_labels 追加）
        self._hw_support: torch.Tensor | None = None
        self._hw_index:   list[dict]          = []

        # 合并后的工作支撑集（recognize 使用这个）
        self._rebuild_merged()

        print(f"Recognizer ready | device={self.device} | "
              f"synthetic={len(self._syn_index)} | "
              f"handwriting={len(self._hw_index)}")

    # ── 手写支撑集加载 ────────────────────────────────────────────────

    def load_handwriting_labels(self, labels_json: str, crops_dir: str) -> int:
        """
        从 handwriting_labels.json + 裁剪图目录提取特征，
        追加到手写支撑集（同一 Unicode 多张取均值）。
        返回成功加载的唯一字符数。
        """
        with open(labels_json, encoding="utf-8") as f:
            data = json.load(f)

        crops_path = Path(crops_dir)
        uid_feats: dict[str, list[torch.Tensor]] = defaultdict(list)
        uid_meta:  dict[str, dict]               = {}

        for item in data["labels"]:
            img_path = crops_path / item["filename"]
            if not img_path.exists():
                continue
            feat = self._extract(Image.open(img_path))
            uid  = item["uid"]
            uid_feats[uid].append(feat)
            uid_meta[uid] = {
                "unicode_id": uid,
                "char":       item["char"],
                "codepoint":  hex(int(uid[1:], 16)),
                "available":  True,
                "source":     "handwriting",
            }

        if not uid_feats:
            return 0

        new_feats = [torch.stack(v).mean(0) for v in uid_feats.values()]
        new_index = list(uid_meta.values())

        # 追加（避免重复：先去掉已有同 Unicode 的手写向量）
        existing_uids = {e["unicode_id"] for e in self._hw_index}
        for feat, entry in zip(new_feats, new_index):
            uid = entry["unicode_id"]
            if uid in existing_uids:
                # 更新已有向量（替换）
                pos = next(i for i, e in enumerate(self._hw_index)
                           if e["unicode_id"] == uid)
                hw_list = self._hw_support.cpu().unbind(0) if \
                          self._hw_support is not None else []
                hw_list = list(hw_list)
                hw_list[pos] = feat
                self._hw_support = torch.stack(hw_list)
                self._hw_index[pos] = entry
            else:
                self._hw_index.append(entry)
                existing_uids.add(uid)
                if self._hw_support is None:
                    self._hw_support = feat.unsqueeze(0)
                else:
                    self._hw_support = torch.cat(
                        [self._hw_support, feat.unsqueeze(0)], dim=0)

        self._rebuild_merged()
        print(f"手写支撑集已更新: {len(self._hw_index)} 个字符 "
              f"(来自 {labels_json})")
        return len(uid_feats)

    def scan_and_load_handwriting(self, search_root: str = ".") -> int:
        """
        递归扫描 search_root 下所有 handwriting_labels.json，
        自动推断 crops 目录并加载。返回总共加载的唯一字符数。
        """
        root   = Path(search_root)
        loaded = 0
        for lf in sorted(root.rglob("handwriting_labels.json")):
            # crops 目录推断：与 labels json 同级，或同级有 *_crops 子目录
            candidates = list(lf.parent.glob("*_crops"))
            crops_dir  = candidates[0] if candidates else lf.parent
            try:
                n = self.load_handwriting_labels(str(lf), str(crops_dir))
                loaded += n
            except Exception as e:
                print(f"加载 {lf} 失败: {e}")
        return loaded

    # ── 内部工具 ─────────────────────────────────────────────────────

    def _extract(self, img: Image.Image) -> torch.Tensor:
        t = _TRANSFORM(img.convert("L").convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.forward_one(t).squeeze(0).cpu()

    def _rebuild_merged(self):
        """合并合成支撑集和手写支撑集，重建工作支撑集。"""
        if self._hw_support is not None and len(self._hw_index) > 0:
            self.support = torch.cat(
                [self._syn_support, self._hw_support], dim=0
            ).to(self.device)
            self.index = self._syn_index + self._hw_index
        else:
            self.support = self._syn_support.to(self.device)
            self.index   = self._syn_index

    @property
    def hw_count(self) -> int:
        return len(self._hw_index)

    # ── 识别 ─────────────────────────────────────────────────────────

    def recognize(self, img: Image.Image, top_k: int = 5) -> list[dict]:
        """
        识别单个字符图像。
        Args:
            img: PIL Image（灰度或 RGB，任意尺寸）
            top_k: 返回前 K 个候选
        Returns:
            list of {rank, unicode_id, char, codepoint, distance, confidence, source}
        """
        query = self._extract(img).to(self.device)
        dists = torch.cdist(query.unsqueeze(0), self.support, p=1).squeeze(0)

        k = min(top_k, len(self.index))
        top_dists, top_idx = torch.topk(dists, k=k, largest=False)

        results = []
        for rank, (idx, dist) in enumerate(
                zip(top_idx.tolist(), top_dists.tolist()), 1):
            entry      = self.index[idx]
            confidence = 1.0 / (1.0 + dist / 100.0)
            results.append({
                "rank":       rank,
                "unicode_id": entry["unicode_id"],
                "char":       entry["char"],
                "codepoint":  entry["codepoint"],
                "distance":   round(dist, 4),
                "confidence": round(confidence, 4),
                "source":     entry.get("source", "synthetic"),
            })
        return results

