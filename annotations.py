"""Persist human annotations separately from model predictions."""
import hashlib
import json
import math
import threading
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageOps
from panel_assistant import EXTENSIONS, write_json


class AnnotationStore:
    def __init__(self, images, output):
        self.images, self.output = Path(images), Path(output)
        self.lock = threading.Lock()

    def pages(self):
        result = []
        for path in sorted(self.images.rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                relative = path.relative_to(self.images).as_posix()
                key = hashlib.sha256(relative.encode()).hexdigest()[:24]
                annotation = self.load(key)
                development = path.stem.endswith(tuple(f'_{i:02d}' for i in (3,6,8,9,11,12)))
                result.append(dict(id=key, name=relative, development=development,
                    split=annotation.get('split','dev' if development else 'test'),
                    saved=bool(annotation), verified=annotation.get('verified',False)))
        return result

    def source(self, key):
        if not isinstance(key,str) or len(key)!=24:
            raise ValueError('无效图片编号')
        for path in sorted(self.images.rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                relative=path.relative_to(self.images).as_posix()
                if hashlib.sha256(relative.encode()).hexdigest()[:24]==key:
                    return path
        raise ValueError('图片不存在')

    def load(self, key):
        if not isinstance(key,str) or len(key)!=24 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('无效图片编号')
        path=self.output/'annotations'/(key+'.json')
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def save(self, payload):
        key=payload.get('id')
        source=self.source(key)
        split=payload.get('split')
        if split not in ('dev','test'):
            raise ValueError('请选择开发集或测试集')
        if source.stem.endswith(tuple(f'_{i:02d}' for i in (3,6,8,9,11,12))) and split!='dev':
            raise ValueError('已试跑的开发页不能转为保留测试页')
        verified=payload.get('verified')
        if type(verified) is not bool:
            raise ValueError('核验状态必须为布尔值')
        boxes=payload.get('boxes')
        if not isinstance(boxes,list) or len(boxes)>500:
            raise ValueError('框列表无效（最多500个）')
        with Image.open(source) as raw:
            image=ImageOps.exif_transpose(raw)
            width,height=image.size
        clean=[]
        for box in boxes:
            if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in box):
                raise ValueError('框须为四个有限数值 xywh')
            x,y,w,h=[round(v) for v in box]
            if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height:
                raise ValueError('框必须在图片内且有正面积')
            clean.append([x,y,w,h])
        notes=payload.get('notes','')
        if not isinstance(notes,str) or len(notes)>2000:
            raise ValueError('备注最多2000字')
        record=dict(id=key,source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            image_size=[width,height], bbox_format='xywh', coordinate_system='EXIF-oriented pixels',
            boxes=clean,verified=verified,split=split,notes=notes,annotation_method='manual_from_blank',
            updated_at=datetime.now().astimezone().isoformat())
        with self.lock:
            folder=self.output/'annotations'
            folder.mkdir(parents=True,exist_ok=True)
            history=folder/'history'
            history.mkdir(exist_ok=True)
            write_json(history/(key+'_'+uuid.uuid4().hex+'.json'),record)
            target=folder/(key+'.json')
            temporary=target.with_suffix('.tmp')
            write_json(temporary,record)
            temporary.replace(target)
            all_records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('*.json'))]
            for group in ('dev','test'):
                # Only explicitly verified pages are eligible for evaluation.
                selected=[r for r in all_records if r['split']==group and r['verified']]
                target=self.output/(group+'_truth.json')
                temporary=target.with_suffix('.tmp')
                write_json(temporary,dict(pages=selected))
                temporary.replace(target)
        return record
