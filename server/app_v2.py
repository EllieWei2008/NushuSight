# -*- coding: utf-8 -*-
"""
app_v2.py — 女书 OCR Studio v2
单页面完整流程：上传 → 检测+纠错 → 识别+人工选择 → 翻译 → 导出

用法:
  cd C:\\Users\\I504158\\nushu-ocr
  py demo/app_v2.py --port 5000

访问: http://127.0.0.1:5000
"""

import sys, io, os, json, base64, argparse, datetime
from pathlib import Path

# 强制 UTF-8 输出（Windows）
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / 'phase1'))

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from PIL import Image
import cv2
import numpy as np

from recognize import NushuRecognizer
from detect    import detect_chars

# 直接复制 normalize_crop 函数，避免 crop_boxes.py 的 stdout 重定向冲突
import cv2
import numpy as np

TARGET_SIZE = 64  # 与 test_date 手写图一致（旧逻辑）

def normalize_crop(gray: np.ndarray):
    """保留：供其他模块调用的兼容接口。"""
    return crop_and_normalize_gray(gray)

def crop_and_normalize_gray(gray: np.ndarray):
    """灰度图 → 正方形白边居中 → CLAHE → 64×64。与 test_date 手写图生成逻辑一致。"""
    h, w = gray.shape
    side = max(h, w)
    square = np.ones((side, side), dtype=np.uint8) * 255
    oy = (side - h) // 2; ox = (side - w) // 2
    square[oy:oy+h, ox:ox+w] = gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(square)
    result = cv2.resize(enhanced, (TARGET_SIZE, TARGET_SIZE),
                        interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(result)

app = Flask(__name__)

_MODEL_PATH      = str(_HERE.parent / 'checkpoints' / 'oneshot' / 'oneshot_best.pth')
_DEMO_DIR        = str(_HERE)
_CORRECTIONS_DIR = str(_HERE.parent / 'corrections')
_recognizer: NushuRecognizer = None

# ── 全局资源 ──────────────────────────────────────────────────────────────────

def get_recognizer() -> NushuRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = NushuRecognizer(_MODEL_PATH, _DEMO_DIR)
    return _recognizer


def load_zhiku() -> dict:
    p = _HERE / 'zhiku.json'
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return {}

_ZHIKU: dict = load_zhiku()


def load_pua() -> dict:
    p = _HERE / 'pua_chars.json'
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        # 过滤掉 _comment / _format / _next_uid 等元字段
        return {k: v for k, v in data.items()
                if not k.startswith('_') and isinstance(v, dict)}
    return {}

def save_pua(pua: dict):
    p = _HERE / 'pua_chars.json'
    # 读取现有文件保留元字段
    existing = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    meta = {k: v for k, v in existing.items() if k.startswith('_')}
    # 计算下一个可用码位
    used = [int(k[1:], 16) for k in pua if k.startswith('U1B3')]
    next_cp = max(used) + 1 if used else 0x1B300
    meta['_next_uid'] = f'U{next_cp:05X}'.upper()
    merged = {**meta, **pua}
    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                 encoding='utf-8')

_PUA: dict = load_pua()


def load_lookalike() -> dict:
    """加载形近字组，返回 uid -> group(list) 反查索引。"""
    p = _HERE / 'lookalike_groups.json'
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding='utf-8'))
    groups = data.get('groups', [])
    index: dict = {}
    for g in groups:
        uids = g.get('uids', [])
        for uid in uids:
            index[uid.upper()] = uids
    return index

_LOOKALIKE: dict = load_lookalike()


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def nushu_sort(boxes: list) -> list:
    """
    按女书阅读顺序排序并编号：列从右到左，列内从上到下。
    同列判定：两框 x 中心差 < 较大框宽度的 60%。
    """
    if not boxes:
        return []
    # 按 x 中心从大到小（右→左）
    sorted_boxes = sorted(boxes, key=lambda b: -(b['x'] + b['w'] / 2))
    cols = []
    for b in sorted_boxes:
        cx = b['x'] + b['w'] / 2
        placed = False
        for col in cols:
            col_cx = sum(c['x'] + c['w'] / 2 for c in col) / len(col)
            threshold = max(b['w'], col[0]['w']) * 0.6
            if abs(cx - col_cx) < threshold:
                col.append(b)
                placed = True
                break
        if not placed:
            cols.append([b])
    # 列内按 y 从小到大（上→下），重新编号
    n = 1
    result = []
    for col in cols:
        for b in sorted(col, key=lambda b: b['y']):
            result.append({**b, 'n': n})
            n += 1
    return result


def crop_and_normalize(img_bgr: np.ndarray, box: dict) -> Image.Image:
    """
    按四顶点多边形精确裁剪：
    1. 取外接矩形（加3px防截断）
    2. 多边形外填白（遮掉相邻字符）
    3. 构建正方形，CLAHE增强，resize到64×64
    """
    H, W = img_bgr.shape[:2]
    x, y, w, h = int(box['x']), int(box['y']), int(box['w']), int(box['h'])
    pad = 3
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(W, x + w + pad); y2 = min(H, y + h + pad)
    crop_bgr  = img_bgr[y1:y2, x1:x2].copy()
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # 多边形遮盖
    if 'quad' in box and box['quad']:
        ch, cw = crop_gray.shape
        pts = np.array([[int(p[0])-x1, int(p[1])-y1] for p in box['quad']],
                       dtype=np.int32)
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        crop_gray[mask == 0] = 255
    return crop_and_normalize_gray(crop_gray)


# ── API 路由 ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    html_path = _HERE / 'ocr_studio.html'
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return '<h2>ocr_studio.html not found</h2>', 404


@app.route('/nushu_chars.json')
def serve_nushu_chars():
    """供前端加载女书字符字典。"""
    p = _HERE / 'nushu_chars.json'
    if p.exists():
        return p.read_text(encoding='utf-8'), 200, {'Content-Type': 'application/json; charset=utf-8'}
    return jsonify([]), 200


@app.route('/detect', methods=['POST'])
def api_detect():
    """接受图片文件，返回自动检测的字符框列表。"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file          = request.files['image']
    min_size      = request.form.get('min_size')
    max_size      = request.form.get('max_size')
    col_threshold = request.form.get('col_threshold')
    min_size      = int(min_size) if min_size and int(min_size) > 0 else None
    max_size      = int(max_size) if max_size and int(max_size) > 0 else None
    col_threshold = float(col_threshold) if col_threshold else 0.6

    tmp = _HERE / 'output' / '_detect_tmp.png'
    tmp.parent.mkdir(exist_ok=True)
    file.save(str(tmp))

    try:
        detected = detect_chars(str(tmp), min_size=min_size, max_size=max_size,
                                col_threshold=col_threshold)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    boxes = [{'n': det['order']+1, 'x': det['bbox'][0], 'y': det['bbox'][1],
               'w': det['bbox'][2], 'h': det['bbox'][3]}
             for det in detected]

    # 女书阅读顺序
    boxes = nushu_sort(boxes)
    return jsonify({'boxes': boxes, 'total': len(boxes)})


@app.route('/recognize_boxes', methods=['POST'])
def api_recognize_boxes():
    """
    接受图片 + 框列表，返回每框的 Top-5 识别结果 + 翻译。
    Body (multipart):
      image: 图片文件
      boxes: JSON 字符串，[{n, x, y, w, h}, ...]
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file  = request.files['image']
    boxes = json.loads(request.form.get('boxes', '[]'))
    # 按编号排序，确保手工插入的框顺序正确
    boxes = sorted(boxes, key=lambda b: b.get('n', 0))

    tmp = _HERE / 'output' / '_recog_tmp.png'
    tmp.parent.mkdir(exist_ok=True)
    file.save(str(tmp))

    img_bgr = cv2.imread(str(tmp))
    if img_bgr is None:
        return jsonify({'error': 'Cannot read image'}), 400

    r       = get_recognizer()
    results = []
    crops_debug_dir = _HERE / 'output' / 'debug_crops'
    crops_debug_dir.mkdir(exist_ok=True)

    for box in boxes:
        crop_pil  = crop_and_normalize(img_bgr, box)
        # 保存调试裁剪图
        crop_pil.save(str(crops_debug_dir / f"box_{box['n']:03d}.png"))
        crop_b64  = pil_to_b64(crop_pil)
        preds     = r.recognize(crop_pil, top_k=5)

        # 追加翻译
        for p in preds:
            uid  = p['unicode_id']
            info = _ZHIKU.get(uid, {})
            p['hanzi']  = info.get('hanzi',  '')
            p['pinyin'] = info.get('pinyin', '')

        results.append({
            'n':           box['n'],
            'bbox':        [box['x'], box['y'], box['w'], box['h']],
            'crop_b64':    crop_b64,
            'predictions': preds,
        })

    return jsonify({'chars': results})


@app.route('/zhiku/<uid>')
def api_zhiku(uid: str):
    """查询单个字符的汉字翻译（标准字库 + PUA 字库）。"""
    uid  = uid.upper()
    # 先查标准字库
    info = _ZHIKU.get(uid)
    if info:
        return jsonify(info)
    # 再查 PUA 字库
    pua_info = _PUA.get(uid)
    if pua_info:
        return jsonify({
            'hanzi':  pua_info.get('hanzi', ''),
            'pinyin': pua_info.get('note', ''),
            'is_pua': True,
            'label':  pua_info.get('label', ''),
            'source': pua_info.get('source', ''),
        })
    return jsonify({'hanzi': '', 'pinyin': ''}), 200


@app.route('/lookalike/<uid>')
def api_lookalike(uid: str):
    """返回 uid 所在形近字组，附带每个字符的汉字信息。"""
    uid = uid.upper()
    group_uids = _LOOKALIKE.get(uid, [])
    if not group_uids:
        return jsonify({'group': []})

    result = []
    for u in group_uids:
        char_cp = int(u[1:], 16)
        char = chr(char_cp)
        info = _ZHIKU.get(u) or _PUA.get(u) or {}
        result.append({
            'uid':   u,
            'char':  char,
            'hanzi': info.get('hanzi', ''),
        })
    return jsonify({'group': result})


@app.route('/pua', methods=['GET'])
def api_pua_list():
    """返回所有 PUA 字符列表。"""
    return jsonify({'chars': _PUA, 'total': len(_PUA)})


@app.route('/pua/add', methods=['POST'])
def api_pua_add():
    """
    新增一个 PUA 字符。
    Body JSON:
      label:    简短名称（如"玉秀特有字1"）
      source:   来源文献（如"YuXiu手稿"）
      hanzi:    对应汉字（若有）
      note:     备注
      added_by: 登记人
    """
    global _PUA
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    # 分配下一个码位
    existing_cps = [int(k[1:], 16) for k in _PUA if k.startswith('U1B3')]
    next_cp = max(existing_cps) + 1 if existing_cps else 0x1B300
    uid     = f'U{next_cp:05X}'.upper()

    _PUA[uid] = {
        'label':      data.get('label', f'未命名字符-{uid}'),
        'source':     data.get('source', ''),
        'hanzi':      data.get('hanzi', ''),
        'note':       data.get('note', ''),
        'added_by':   data.get('added_by', ''),
        'added_date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'crop_b64':   data.get('crop_b64', ''),
    }
    save_pua(_PUA)
    return jsonify({'uid': uid, 'char': _PUA[uid]})


@app.route('/pua/<uid>', methods=['DELETE'])
def api_pua_delete(uid: str):
    """删除一个 PUA 字符登记。"""
    global _PUA
    uid = uid.upper()
    if uid not in _PUA:
        return jsonify({'error': 'Not found'}), 404
    del _PUA[uid]
    save_pua(_PUA)
    return jsonify({'deleted': uid})


@app.route('/save_corrections', methods=['POST'])
def api_save_corrections():
    """
    保存人工纠错数据到 demo/corrections/。
    Body (JSON):
      source_image: str
      corrections: [{order, bbox, auto_top1, human_choice, was_corrected, crop_b64?}, ...]
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    out_dir = Path(_CORRECTIONS_DIR)
    out_dir.mkdir(exist_ok=True)

    ts      = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    payload = {
        'session':      ts,
        'source_image': data.get('source_image', ''),
        'corrections':  data.get('corrections', []),
    }

    out_file = out_dir / f'corrections_{ts}.json'
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding='utf-8')

    # 只保存人工修正过女书字符的裁剪图（was_corrected=True）
    crops_dir   = out_dir / 'crops' / ts
    saved_crops = 0
    for c in payload['corrections']:
        if c.get('was_corrected') and c.get('crop_b64') and c.get('human_choice'):
            crops_dir.mkdir(parents=True, exist_ok=True)
            uid      = c['human_choice']
            img_data = base64.b64decode(c['crop_b64'])
            fname    = crops_dir / f"{uid}_corr_{c['order']:03d}.png"
            fname.write_bytes(img_data)
            saved_crops += 1

    return jsonify({
        'saved':       str(out_file),
        'crops_saved': saved_crops,
    })


@app.route('/debug')
def debug():
    r = get_recognizer()
    return jsonify({
        'model_path':    _MODEL_PATH,
        'support_shape': list(r.support.shape),
        'hw_count':      r.hw_count,
        'zhiku_size':    len(_ZHIKU),
    })


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    global _MODEL_PATH, _DEMO_DIR, _ZHIKU, _LOOKALIKE

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    default=str(_HERE.parent / 'checkpoints' / 'oneshot' / 'oneshot_best.pth'))
    parser.add_argument('--demo_dir', default=str(_HERE))
    parser.add_argument('--host',     default='127.0.0.1')
    parser.add_argument('--port',     type=int, default=5000)
    parser.add_argument('--debug',    action='store_true', help='开启 Flask debug 模式（显示详细错误）')
    args = parser.parse_args()

    _MODEL_PATH = args.model
    _DEMO_DIR   = args.demo_dir
    _ZHIKU      = load_zhiku()
    _LOOKALIKE  = load_lookalike()

    print(f'字库加载: {len(_ZHIKU)} 条')
    get_recognizer()   # 预热

    print(f'\nOCR Studio v2 running at http://{args.host}:{args.port}')
    print('Press Ctrl+C to stop.\n')
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
