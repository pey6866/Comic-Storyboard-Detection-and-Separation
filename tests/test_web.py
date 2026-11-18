import json
import os
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer
from PIL import Image, ImageDraw
from web_server import Handler, WorkflowAgent
from panel_assistant import ROOT


class WebTests(unittest.TestCase):
    @unittest.skipUnless(Path(os.environ.get('PANEL_MODEL_PATH', ROOT/'models/best.onnx')).is_file(),
                         'Download models/best.onnx to run the real ONNX web integration test')
    def test_upload_preview_download_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            server.agent = WorkflowAgent(Path(tmp))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                def upload(content, origin=None):
                    headers = {'Content-Type': 'application/octet-stream'}
                    if origin:
                        headers['Origin'] = origin
                    return urllib.request.urlopen(urllib.request.Request(base+'/api/detect', data=content, headers=headers))
                with urllib.request.urlopen(base) as response:
                    self.assertIn('灰度'.encode(), response.read())
                with self.assertRaises(urllib.error.HTTPError) as error:
                    upload(b'not an image')
                self.assertEqual(error.exception.code, 400)
                with self.assertRaises(urllib.error.HTTPError) as error:
                    upload(b'x', 'http://foreign.example')
                self.assertEqual(error.exception.code, 403)
                source = Path(tmp)/'upload_sample.png'
                sample = Image.new('RGB', (800, 1200), 'white')
                draw = ImageDraw.Draw(sample)
                for box in [(40,40,760,370),(40,410,370,1150),(410,410,760,1150)]:
                    draw.rectangle(box, outline='black', width=8)
                sample.save(source)
                with upload(source.read_bytes()) as response:
                    result = json.load(response)
                self.assertGreater(result['count'], 0)
                for name, url in result['files'].items():
                    with urllib.request.urlopen(base+url+'?download=1') as response:
                        self.assertIn('attachment', response.headers['Content-Disposition'])
                        body = response.read()
                    if name == 'boundaries':
                        boundaries = json.loads(body)
                        self.assertEqual(len(boundaries['panels']), result['count'])
                        self.assertFalse(boundaries['human_verified'])
                    else:
                        import io
                        with Image.open(io.BytesIO(body)) as image:
                            self.assertEqual(list(image.size), result['image_size'])
                            if name == 'grayscale':
                                self.assertEqual(image.mode, 'L')
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base+'/files/../../models/best.onnx')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
