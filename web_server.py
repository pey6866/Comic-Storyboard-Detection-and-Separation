"""Local upload UI + task workflow API. Run with .venv/Scripts/python.exe web_server.py."""
import argparse
import os
import io
import json
import re
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from annotations import AnnotationStore

from PIL import Image, ImageOps, UnidentifiedImageError
from panel_assistant import ROOT, Inkwell, process, write_json

MAX_UPLOAD = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 30_000_000


class WorkflowAgent:
    """Validate -> detect -> check -> export. Model initialized once per process."""
    def __init__(self, output):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.detector = None

    def run(self, content):
        try:
            with Image.open(io.BytesIO(content)) as raw:
                if raw.format not in {'JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF'}:
                    raise ValueError('仅支持 JPG、PNG、WebP、BMP、TIFF 图片。')
                if raw.width * raw.height > Image.MAX_IMAGE_PIXELS:
                    raise ValueError('图片超过 3000 万像素，请缩小后上传。')
                image = ImageOps.exif_transpose(raw).convert('RGB')
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError('无法读取图片，请检查文件是否损坏。') from exc
        task_id = uuid.uuid4().hex
        task = self.output / task_id
        task.mkdir()
        source = task/'upload.png'
        image.save(source)
        with self.lock:
            if self.detector is None:
                self.detector = Inkwell(Path(os.environ.get('PANEL_MODEL_PATH', ROOT/'models/best.onnx')), .25, .45)
            record = process(source, task, self.detector, 'inkwell',
                             dict(confidence=.25, nms=.45, review_threshold=.5))
        if record['status'] == 'failed':
            raise RuntimeError('检测失败，请查看本地任务记录。')
        folder = task/record['page_id']
        # Expose only portable image geometry; keep local paths in the internal log.
        boundaries = dict(schema_version=1, task_id=task_id, image_size=record['image_size'],
            bbox_format='xywh', coordinate_system=record['coordinate_system'],
            detector=record['detector'], model_revision=record['model_revision'],
            parameters=record['parameters'], status=record['status'], warnings=record['warnings'],
            human_verified=False, panels=[{k:p[k] for k in ('id','bbox','confidence')} for p in record['panels']])
        write_json(folder/'boundaries.json', boundaries)
        files = {name:f'/files/{task_id}/{record["page_id"]}/{filename}' for name, filename in
                 dict(gray='preview_gray.png', color='preview.png', grayscale='grayscale.png', boundaries='boundaries.json').items()}
        return dict(task_id=task_id, image_size=record['image_size'], count=len(record['panels']),
                    status=record['status'], warnings=record['warnings'], elapsed_ms=record['elapsed_ms'],
                    panels=boundaries['panels'], files=files,
                    steps=['输入校验', '模型检测', '边界检查', '预览与数据导出'])


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, content, mime='application/json; charset=utf-8', attachment=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        if attachment:
            self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(content)

    def json_response(self, status, data):
        self.respond(status, json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == '/annotate':
            return self.respond(200, (ROOT/'web/annotate.html').read_bytes(), 'text/html; charset=utf-8')
        if route == '/api/pages':
            return self.json_response(200, self.server.annotations.pages())
        if route in ('/api/page-image','/api/annotation'):
            try:
                key=parse_qs(urlsplit(self.path).query).get('id',[''])[0]
                source=self.server.annotations.source(key)
                if route=='/api/annotation':
                    return self.json_response(200,self.server.annotations.load(key))
                with Image.open(source) as raw:
                    image=ImageOps.exif_transpose(raw).convert('RGB')
                    buffer=io.BytesIO()
                    image.save(buffer,format='PNG')
                return self.respond(200,buffer.getvalue(),'image/png')
            except ValueError as exc:
                return self.json_response(400,dict(error=str(exc)))
        if route in ('/api/truth/dev','/api/truth/test'):
            group=route.rsplit('/',1)[1]
            path=self.server.annotations.output/(group+'_truth.json')
            return self.respond(200,path.read_bytes() if path.exists() else b'{"pages":[]}',
                                attachment=group+'_truth.json')
        if route == '/':
            return self.respond(200, (ROOT/'web/index.html').read_bytes(), 'text/html; charset=utf-8')
        if route == '/api/health':
            return self.json_response(200, dict(status='ready', model_exists=(ROOT/'models/best.onnx').exists()))
        match = re.fullmatch(r'/files/([a-f0-9]{32})/(upload_[a-f0-9]{12})/(preview.png|preview_gray.png|grayscale.png|boundaries.json)', route)
        if match:
            task_id, page, filename = match.groups()
            path = self.server.agent.output/task_id/page/filename
            if path.is_file():
                mime = 'application/json; charset=utf-8' if filename.endswith('.json') else 'image/png'
                return self.respond(200, path.read_bytes(), mime, filename if 'download=1' in urlsplit(self.path).query else None)
        self.json_response(404, dict(error='文件不存在。'))

    def do_POST(self):
        if self.path not in ('/api/detect','/api/annotation'):
            return self.json_response(404, dict(error='接口不存在。'))
        # Browser requests must originate from this same local UI.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.json_response(403, dict(error='请从本地页面上传。'))
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            size = 0
        if not 0 < size <= MAX_UPLOAD:
            return self.json_response(413, dict(error='请选择非空且不超过 20 MB 的图片。'))
        self.connection.settimeout(60)
        try:
            content = self.rfile.read(size)
            if len(content) != size:
                raise ValueError('图片上传不完整，请重试。')
            result = (self.server.annotations.save(json.loads(content)) if self.path=='/api/annotation'
                      else self.server.agent.run(content))
            self.json_response(200, result)
        except (ValueError, Image.DecompressionBombError) as exc:
            self.json_response(400, dict(error=str(exc)))
        except Exception:
            traceback.print_exc()
            self.json_response(500, dict(error='检测未完成，请检查权重文件和终端日志。'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8877)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.agent = WorkflowAgent(ROOT/'outputs/web')
    server.annotations = AnnotationStore(ROOT/'漫画', ROOT/'evaluation')
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
