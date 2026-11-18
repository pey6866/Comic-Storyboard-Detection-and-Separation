"""Comic panel detection CLI. Coordinates always refer to EXIF-oriented pixels."""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def draw_panels(image, panels, grayscale=False):
    """Use luminance-neutral backgrounds plus contrasting halos for review."""
    preview = ImageOps.grayscale(image).convert('RGB') if grayscale else image.copy()
    draw = ImageDraw.Draw(preview)
    width = max(4, round(min(image.size) / 150))
    font_size = max(18, round(min(image.size) / 35))
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()
    colors = ['#e6004c', '#0063ff', '#00854a', '#9b19df', '#cc5800']
    for panel in panels:
        x, y, w, h = panel['bbox']
        color = colors[(panel['id']-1) % len(colors)]
        rect = (x, y, x+w-1, y+h-1)
        draw.rectangle(rect, outline='white', width=width+4)
        draw.rectangle(rect, outline=color, width=width)
        label = str(panel['id'])
        bounds = draw.textbbox((0, 0), label, font=font)
        tw, th = bounds[2]-bounds[0]+12, bounds[3]-bounds[1]+10
        lx, ly = min(x+width, max(0, image.width-tw)), min(y+width, max(0, image.height-th))
        draw.rectangle((lx, ly, lx+tw, ly+th), fill=color, outline='white', width=2)
        draw.text((lx+6, ly+5-bounds[1]), label, fill='white', font=font)
    return preview


def iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0, right-x) * max(0, bottom-y)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0


def letterbox(image, size):
    w, h = image.size
    ratio = min(size/w, size/h)
    nw, nh = round(w*ratio), round(h*ratio)
    left, top = (size-nw)//2, (size-nh)//2
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    blob = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return blob, (nw/w, nh/h, left, top)


def restore(box, transform, image_size):
    sx, sy, left, top = transform
    cx, cy, w, h = box
    iw, ih = image_size
    x1 = max(0, min(iw, math.floor((cx-w/2-left)/sx)))
    y1 = max(0, min(ih, math.floor((cy-h/2-top)/sy)))
    x2 = max(0, min(iw, math.ceil((cx+w/2-left)/sx)))
    y2 = max(0, min(ih, math.ceil((cy+h/2-top)/sy)))
    return [x1, y1, x2-x1, y2-y1]


class Inkwell:
    def __init__(self, model, confidence, nms):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError('Unsupported model input: ' + str(shape))
        self.size = shape[2] if isinstance(shape[2], int) else 640
        if isinstance(shape[3], int) and shape[3] != self.size:
            raise ValueError('Expected square model input: ' + str(shape))
        self.confidence, self.nms = confidence, nms
        self.metadata = {'sha256': hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                         'providers': self.session.get_providers(), 'input_shape': shape,
                         'custom_metadata': self.session.get_modelmeta().custom_metadata_map}

    def detect(self, image):
        blob, transform = letterbox(image, self.size)
        output = self.session.run(None, {self.input.name: blob})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 5:
            raise ValueError('Expected raw single-class YOLO [1,5,N], got ' + str(output.shape))
        candidates = []
        for row in output[0].T:
            if not np.isfinite(row).all() or row[4] < self.confidence:
                continue
            box = restore(row[:4], transform, image.size)
            if box[2] > 0 and box[3] > 0:
                candidates.append((box, float(row[4])))
        kept = []
        for box, score in sorted(candidates, key=lambda item: -item[1]):
            if all(iou(box, previous[0]) <= self.nms for previous in kept):
                kept.append((box, score))
        return kept


class Baseline:
    metadata = {'method': 'OpenCV Otsu dark contours; min area 2%; no rotation', 'confidence': None}

    def detect(self, image):
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = image.width * image.height
        boxes = [cv2.boundingRect(c) for c in contours]
        return [(list(b), None) for b in boxes if b[2]*b[3] >= area*.02 and min(b[2:]) >= 10]


def process(source, destination, detector, name, parameters):
    started = time.perf_counter()
    page_id = source.stem + '_' + hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]
    folder = destination / page_id
    folder.mkdir()
    record = dict(source=str(source.resolve()), page_id=page_id, image_size=None,
                  coordinate_system='EXIF-oriented RGB pixels; xywh; id is not reading order',
                  detector=name, model_revision=detector.metadata, parameters=parameters,
                  status='processing', warnings=[], panels=[])
    record['human_verified'] = False
    try:
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        record['image_size'] = list(image.size)
        start = time.perf_counter()
        results = detector.detect(image)
        record['inference_ms'] = (time.perf_counter()-start)*1000
        for index, (box, score) in enumerate(results, 1):
            x, y, w, h = box
            crop = f'panel_{index:03d}.png'
            image.crop((x, y, x+w, y+h)).save(folder / crop)
            record['panels'].append(dict(id=index, bbox=box, confidence=score, crop_file=crop))
        if not results:
            record['warnings'].append('no_detections')
        if any(s is not None and s < parameters['review_threshold'] for _, s in results):
            record['warnings'].append('low_confidence')
        if any(min(box[2:]) < 4 or max(box[2:])/min(box[2:]) > 50 for box, _ in results):
            record['warnings'].append('suspicious_thin_box')
        if name == 'baseline':
            record['warnings'].append('baseline_requires_review')
        record['status'] = 'needs_review' if record['warnings'] else 'success'
        draw_panels(image, record['panels']).save(folder / 'preview.png')
        draw_panels(image, record['panels'], grayscale=True).save(folder / 'preview_gray.png')
        ImageOps.grayscale(image).save(folder / 'grayscale.png')
        record['preview_files'] = {'color': 'preview.png', 'gray': 'preview_gray.png', 'grayscale': 'grayscale.png'}
    except Exception as error:
        record['status'] = 'failed'
        record['warnings'].append(type(error).__name__ + ': ' + str(error))
    record['elapsed_ms'] = (time.perf_counter()-started)*1000
    write_json(folder / 'result.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    parser.add_argument('--detector', choices=['inkwell', 'baseline'], default='inkwell')
    parser.add_argument('--model', type=Path, default=ROOT/'models/best.onnx')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--nms', type=float, default=.45)
    parser.add_argument('--review-threshold', type=float, default=.5)
    args = parser.parse_args()
    for key in ('confidence', 'nms', 'review_threshold'):
        if not 0 <= getattr(args, key) <= 1:
            parser.error(key + ' must be in [0,1]')
    if not args.input.exists():
        parser.error('Input does not exist: ' + str(args.input))
    files = sorted(p for p in args.input.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file()) if args.input.is_dir() else [args.input]
    if not files:
        parser.error('No supported images found')
    start = time.perf_counter()
    try:
        detector = Inkwell(args.model, args.confidence, args.nms) if args.detector == 'inkwell' else Baseline()
    except Exception as error:
        parser.exit(2, 'Detector initialization failed: ' + str(error) + '\n')
    load_ms = (time.perf_counter()-start)*1000
    destination = args.output / (datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True)
    parameters = {key: getattr(args, key) for key in ('confidence', 'nms', 'review_threshold')}
    records = [process(p, destination, detector, args.detector, parameters) for p in files]
    write_json(destination/'run.json', dict(model_load_ms=load_ms, pages=records,
               timing='inference_ms includes preprocessing and NMS; elapsed_ms includes image IO and export; model load separate'))
    print(str(destination.resolve()))
    for record in records:
        print(record['page_id'], record['status'], len(record['panels']))
    return 1 if any(r['status'] == 'failed' for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
