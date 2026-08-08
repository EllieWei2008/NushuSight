# -*- coding: utf-8 -*-
"""
crop_boxes.py — 根据 annotate.html 导出的 JSON 裁剪 + 归一化 + 识别

用法:
  py demo/crop_boxes.py --json cunxiao-he_boxes.json --image demo/cunxiao-he.jpg
  py demo/crop_boxes.py --json cunxiao-he_boxes.json          # image 路径从 JSON 里读
"""

import sys, io, json, argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase1"))

from recognize import NushuRecognizer

TARGET = 64


def normalize_crop(gray: np.ndarray) -> Image.Image:
    """灰度裁剪图 → 64×64 白底黑字，与训练数据风格一致。"""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=8
    )
    coords = cv2.findNonZero(binary)
    if coords is None:
        return Image.new("L", (TARGET, TARGET), 255)

    x, y, w, h = cv2.boundingRect(coords)
    pad  = int(max(w, h) * 0.15)
    x1   = max(0, x - pad);  y1 = max(0, y - pad)
    x2   = min(gray.shape[1], x + w + pad)
    y2   = min(gray.shape[0], y + h + pad)
    ink  = binary[y1:y2, x1:x2]

    side   = max(ink.shape[0], ink.shape[1])
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy     = (side - ink.shape[0]) // 2
    ox     = (side - ink.shape[1]) // 2
    canvas[oy:oy+ink.shape[0], ox:ox+ink.shape[1]] = ink

    result = cv2.bitwise_not(canvas)
    return Image.fromarray(cv2.resize(result, (TARGET, TARGET),
                                      interpolation=cv2.INTER_LANCZOS4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",      required=True, help="annotate.html 导出的 JSON 文件")
    parser.add_argument("--image",     default=None,  help="原图路径（默认从 JSON 的 image 字段推断）")
    parser.add_argument("--model",     default="./checkpoints/siamese_best.pth")
    parser.add_argument("--demo_dir",  default="./demo")
    parser.add_argument("--out_dir",   default=None,  help="裁剪图输出目录（默认同 JSON 目录）")
    parser.add_argument("--top_k",     type=int, default=5)
    parser.add_argument("--no_recognize", action="store_true", help="只裁剪不识别")
    args = parser.parse_args()

    json_path = Path(args.json)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 确定图片路径
    if args.image:
        img_path = Path(args.image)
    else:
        img_path = json_path.parent / data["image"]
    if not img_path.exists():
        print(f"ERROR: 图片不存在: {img_path}")
        sys.exit(1)

    # 输出目录
    out_dir = Path(args.out_dir) if args.out_dir else \
              json_path.parent / (json_path.stem.replace("_boxes", "") + "_crops")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清空旧裁剪图
    for f in out_dir.glob("*.png"):
        f.unlink()

    print(f"\n女书字符裁剪工具")
    print(f"{'─'*55}")
    print(f"原图:   {img_path}")
    print(f"标注:   {len(data['boxes'])} 个框  (来自 {json_path.name})")
    print(f"输出:   {out_dir}")

    img_bgr = cv2.imread(str(img_path))
    H, W    = img_bgr.shape[:2]

    # 裁剪并归一化
    crop_paths = []
    for box in data["boxes"]:
        n  = box["n"]
        x1 = max(0, box["x"]);         y1 = max(0, box["y"])
        x2 = min(W, box["x"]+box["w"]); y2 = min(H, box["y"]+box["h"])

        gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        norm = normalize_crop(gray)

        fname = out_dir / f"char_{n:03d}.png"
        norm.save(fname)
        crop_paths.append((n, fname, box))

    print(f"裁剪完成: {len(crop_paths)} 张 → {out_dir}\n")

    if args.no_recognize:
        return

    # 识别
    recognizer = NushuRecognizer(args.model, args.demo_dir)
    print(f"{'─'*55}")

    results = []
    for n, crop_path, box in crop_paths:
        img_pil = Image.open(crop_path).convert("L")
        preds   = recognizer.recognize(img_pil, top_k=args.top_k)
        top1    = preds[0]
        results.append({"n": n, "box": box, "predictions": preds})

        print(f"字符 {n:2d}  Top-1: {top1['char']} ({top1['unicode_id']})  "
              f"置信度 {top1['confidence']:.4f}")
        print(f"       Top-5: " +
              "  ".join(f"{p['char']}({p['unicode_id']})" for p in preds))
        print()

    # 汇总
    chars = "".join(r["predictions"][0]["char"] for r in results)
    codes = " ".join(r["predictions"][0]["unicode_id"] for r in results)
    print(f"{'─'*55}")
    print(f"识别结果: {chars}")
    print(f"Unicode:  {codes}")

    # 保存结果 JSON
    result_json = out_dir / "recognition_result.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_json}")


if __name__ == "__main__":
    main()
