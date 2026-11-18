import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from annotations import AnnotationStore


class AnnotationTests(unittest.TestCase):
    def test_annotation_http_roundtrip(self):
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer
        from web_server import Handler
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new('RGB',(100,200)).save(root/'p.jpg')
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            server.annotations=AnnotationStore(root,root/'out')
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urllib.request.urlopen(base+'/api/pages') as response:
                    pages=json.load(response)
                key=pages[0]['id']
                body=json.dumps(dict(id=key,split='test',verified=True,boxes=[[1,2,30,40]])).encode()
                request=urllib.request.Request(base+'/api/annotation',data=body,headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request) as response:
                    self.assertTrue(json.load(response)['verified'])
                with urllib.request.urlopen(base+'/api/truth/test') as response:
                    self.assertEqual(json.load(response)['pages'][0]['boxes'],[[1,2,30,40]])
                with urllib.request.urlopen(base+'/api/annotation?id='+key) as response:
                    self.assertEqual(json.load(response)['boxes'],[[1,2,30,40]])
                with urllib.request.urlopen(base+'/api/page-image?id='+key) as response:
                    self.assertEqual(response.headers['Content-Type'],'image/png')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_draft_verify_revise_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            images=root/'images'
            images.mkdir()
            Image.new('RGB',(100,200)).save(images/'page_06.jpg')
            Image.new('RGB',(100,200)).save(images/'page_10.jpg')
            store=AnnotationStore(images,root/'evaluation')
            pages=store.pages()
            key=pages[0]['id']
            payload=dict(id=key,split='dev',boxes=[[1,2,30,40]],verified=False)
            record=store.save(payload)
            self.assertEqual(record['source'],str((images/'page_06.jpg').resolve()))
            self.assertEqual(json.loads((store.output/'dev_truth.json').read_text())['pages'],[])
            payload['verified']=True
            store.save(payload)
            truth=json.loads((store.output/'dev_truth.json').read_text())
            self.assertTrue(truth['pages'][0]['verified'])
            self.assertEqual(truth['pages'][0]['boxes'],[[1,2,30,40]])
            payload['verified']=False
            store.save(payload)
            self.assertEqual(json.loads((store.output/'dev_truth.json').read_text())['pages'],[])
            self.assertEqual(len(list((store.output/'annotations/history').glob('*.json'))),3)
            payload['split']='test'
            with self.assertRaises(ValueError): store.save(payload)
            payload.update(id=pages[1]['id'],verified=True,boxes=[])
            store.save(payload)
            self.assertEqual(len(json.loads((store.output/'test_truth.json').read_text())['pages']),1)

    def test_invalid_boxes_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new('RGB',(100,200)).save(root/'p.jpg')
            store=AnnotationStore(root,root/'out')
            key=store.pages()[0]['id']
            for box in ([0,0,101,10],[0,0,0,1],[float('nan'),0,1,1],[-1,0,2,2]):
                with self.assertRaises(ValueError):
                    store.save(dict(id=key,split='test',verified=True,boxes=[box]))
            with self.assertRaises(ValueError): store.load('../../bad')
