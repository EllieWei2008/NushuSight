# -*- coding: utf-8 -*-
"""
detect.py — 女书字符检测模块 v4
策略：
  1. Otsu 二值化去背景（干净）
  2. x 方向投影找列（女书竖排，列间有明显空隙）
  3. 每列内 y 方向投影找字符分割点
  4. 按女书阅读顺序编号（右→左，列内上→下）
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def detect_chars(image_path: str,
                 debug_dir: str = None,
                 min_size: int = None,
                 max_size: int = None,
                 col_threshold: float = 0.6) -> list[dict]:
    """
    检测图片中的女书字符区域。
    Returns list of {'bbox': (x,y,w,h), 'image': PIL.Image, 'order': int}
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    H, W = img_bgr.shape[:2]

    # ── 1. 二值化 ─────────────────────────────────────────────────────────────
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 小核闭运算：连接笔画内的细小断点，但不扩大太多
    k_close = max(2, min(H, W) // 150)
    ker     = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker, iterations=1)

    # ── 2. x 方向投影，找列边界 ────────────────────────────────────────────────
    x_proj = binary.sum(axis=0) // 255   # 每列的墨水像素数

    # 平滑投影曲线，避免细笔画造成的小缺口
    x_smooth = np.convolve(x_proj, np.ones(max(3, W//80)), mode='same') / max(3, W//80)

    col_threshold = x_smooth.max() * 0.05   # 低于 5% 峰值的列视为空白
    col_segs = _find_segments(x_smooth > col_threshold, min_len=W // 30)

    if not col_segs:
        return []

    # ── 3. 每列内 y 方向投影，找字符边界 ────────────────────────────────────
    candidates = []   # (x, y, w, h)

    for cx1, cx2 in col_segs:
        col_bin  = binary[:, cx1:cx2]
        y_proj   = col_bin.sum(axis=1) // 255

        # 平滑 y 投影
        y_smooth = np.convolve(y_proj, np.ones(max(3, H//80)), mode='same') / max(3, H//80)
        y_thresh = y_smooth.max() * 0.08

        char_segs = _find_segments(y_smooth > y_thresh, min_len=H // 40)

        for cy1, cy2 in char_segs:
            w = cx2 - cx1
            h = cy2 - cy1
            candidates.append((cx1, cy1, w, h))

    if not candidates:
        return []

    # ── 4. 面积过滤 ───────────────────────────────────────────────────────────
    sides  = sorted([max(w, h) for x, y, w, h in candidates])
    median = sides[len(sides) // 2]
    lo = min_size if min_size is not None else max(8,  int(median * 0.3))
    hi = max_size if max_size is not None else int(median * 3.0)

    filtered = []
    for x, y, w, h in candidates:
        side   = max(w, h)
        aspect = side / max(min(w, h), 1)
        if side < lo or side > hi:
            continue
        if aspect > 8:
            continue
        filtered.append((x, y, w, h))

    if not filtered:
        return []

    # ── 5. 女书阅读顺序（右→左，列内上→下）──────────────────────────────────
    filtered = _nushu_sort(filtered, col_threshold=col_threshold)

    # ── 6. 构建结果 ───────────────────────────────────────────────────────────
    results = []
    for order, (x, y, w, h) in enumerate(filtered):
        pad  = int(max(w, h) * 0.10)
        x1   = max(0, x - pad);      y1 = max(0, y - pad)
        x2   = min(W, x + w + pad);  y2 = min(H, y + h + pad)
        crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        results.append({
            "bbox":  (x, y, w, h),
            "image": Image.fromarray(crop),
            "order": order,
        })

    if debug_dir:
        _save_debug(img_bgr, results, binary, x_proj, debug_dir,
                    Path(image_path).stem)

    return results


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _find_segments(mask: np.ndarray, min_len: int = 5) -> list[tuple]:
    """在布尔掩码中找连续 True 的段，返回 [(start, end), ...]，过滤短段。"""
    segs = []
    in_seg = False
    start  = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            in_seg = True
            start  = i
        elif not v and in_seg:
            in_seg = False
            if i - start >= min_len:
                segs.append((start, i))
    if in_seg and len(mask) - start >= min_len:
        segs.append((start, len(mask)))
    return segs


def _nushu_sort(boxes: list, col_threshold: float = 0.6) -> list:
    """女书阅读顺序：列从右到左，列内从上到下。
    col_threshold: 列分组宽松度，值越大允许同一列的字符横向偏移越大。
                   默认 0.6，倾斜书写建议调大到 0.8-1.2。
    """
    sorted_boxes = sorted(boxes, key=lambda b: -(b[0] + b[2] / 2))
    cols = []
    for b in sorted_boxes:
        cx = b[0] + b[2] / 2
        placed = False
        for col in cols:
            col_cx    = sum(c[0] + c[2]/2 for c in col) / len(col)
            threshold = max(b[2], col[0][2]) * col_threshold
            if abs(cx - col_cx) < threshold:
                col.append(b)
                placed = True
                break
        if not placed:
            cols.append([b])
    result = []
    for col in cols:
        result.extend(sorted(col, key=lambda b: b[1]))
    return result


def _save_debug(img_bgr, results, binary, x_proj, debug_dir, stem):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    vis = img_bgr.copy()
    for r in results:
        x, y, w, h = r["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 200), 2)
        cv2.putText(vis, str(r["order"]+1), (x, max(14, y-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imwrite(str(debug_dir / f"{stem}_detected.jpg"), vis)
    cv2.imwrite(str(debug_dir / f"{stem}_binary.jpg"), binary)
