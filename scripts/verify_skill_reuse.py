"""Execute the pinned yolo-inference guidance through the local ONNX adapter.

Development evidence only: no ground truth, no accuracy or speed improvement claim.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
from panel_assistant import ROOT, Inkwell, letterbox, restore, process, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = sorted((ROOT/'漫画').glob('*.jpg'))
    if not any(p.name.endswith('_06.jpg') for p in files) or not any(p.name.endswith('_09.jpg') for p in files):
        raise SystemExit('Historical skill checks require authorized local development pages *_06.jpg and *_09.jpg. See docs/SKILL_REUSE.md; archived evidence is included without images.')
    source_dir = ROOT/'docs/skill_reuse'
    provenance = json.loads((source_dir/'source_manifest.json').read_text(encoding='utf-8-sig'))
    for name, expected in provenance['sha256'].items():
        if digest(source_dir/'upstream'/name) != expected:
            raise ValueError('Skill provenance mismatch: ' + name)
    destination = source_dir/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True)
    evidence = dict(created_at=datetime.now().astimezone().isoformat(),
        skill=provenance, role='development guidance adapted to ONNX Runtime; not runtime skill loading',
        command=[sys.executable, *sys.argv], python=sys.version, platform=platform.platform(),
        dependencies={p:importlib.metadata.version(p) for p in ('numpy','Pillow','onnxruntime','opencv-python-headless')},
        code_sha256={p:digest(ROOT/p) for p in ('panel_assistant.py','scripts/verify_skill_reuse.py')},
        limitations=['No manual ground truth', 'No Precision/Recall/F1', 'No Ultralytics Python Results API invoked'],
        checks=[], cases=[], comparisons=[])

    def check(name, condition, detail):
        evidence['checks'].append(dict(name=name, passed=bool(condition), detail=detail))
        print(('PASS ' if condition else 'FAIL ') + name, flush=True)

    files = sorted((ROOT/'漫画').glob('*.jpg'))
    page6 = next(p for p in files if p.name.endswith('_06.jpg'))
    page9 = next(p for p in files if p.name.endswith('_09.jpg'))
    evidence['inputs'] = [{'source':str(p), 'sha256':digest(p), 'split':'existing development subset'} for p in (page6,page9)]
    start = time.perf_counter()
    detector = Inkwell(ROOT/'models/best.onnx', .25, .45)
    evidence['model_load_ms'] = (time.perf_counter()-start)*1000
    evidence['model'] = detector.metadata
    # Skill: inherit checkpoint image size and check export NMS behavior.
    check('checkpoint_input_size', detector.size == 640, {'actual':detector.size,'checkpoint':detector.metadata['custom_metadata'].get('imgsz')})
    check('raw_output_requires_external_nms', detector.metadata['custom_metadata'].get('end2end') == 'False',
          {'export_args':detector.metadata['custom_metadata'].get('args'), 'note':'YOLOv8 raw [1,5,N] confirmed by successful detect; not YOLO26 NMS-free head'})

    # Skill Sources / Troubleshooting: PIL is RGB; do not pass BGR blindly.
    blob, _ = letterbox(Image.new('RGB',(640,640),(255,0,0)),640)
    check('rgb_nchw_normalization', blob.dtype == np.float32 and blob.shape == (1,3,640,640)
          and np.array_equal(blob[0,:,320,320], [1,0,0]), {'red_pixel_channels':blob[0,:,320,320].tolist()})
    # Skill Results API distinguishes center xywh from corner xyxy: our exported xywh is top-left.
    image = Image.new('RGB',(333,777))
    _, transform = letterbox(image,640)
    sx,sy,left,top = transform
    box = restore([100*sx+left,200*sy+top,100*sx,200*sy],transform,image.size)
    check('letterbox_coordinate_roundtrip', all(abs(a-b)<=1 for a,b in zip(box,[50,100,100,200])),
          {'expected_top_left_xywh':[50,100,100,200], 'actual':box, 'tolerance_pixels':1})

    fixtures = destination/'fixtures'
    fixtures.mkdir()
    Image.new('RGB',(400,600),'white').save(fixtures/'blank.png')
    (fixtures/'corrupt.png').write_bytes(b'intentionally invalid image')
    params=dict(confidence=.25,nms=.45,review_threshold=.5)
    for label, source in [('normal',page6),('low_score',page9),('blank',fixtures/'blank.png'),('corrupt',fixtures/'corrupt.png')]:
        out = destination/label
        out.mkdir()
        record = process(source,out,detector,'inkwell',params)
        evidence['cases'].append(dict(case=label,status=record['status'],warnings=record['warnings'],
            panels=len(record['panels']), elapsed_ms=record['elapsed_ms'], result=str((out/record['page_id']/'result.json').relative_to(destination))))
        if label=='normal':
            check('normal_inference', record['status']!='failed' and len(record['panels'])>0, record['status'])
        elif label=='low_score':
            check('real_low_score_review', 'low_confidence' in record['warnings'] and record['status']=='needs_review', record['warnings'])
        elif label=='blank':
            check('empty_result_review', not record['panels'] and 'no_detections' in record['warnings'],record['warnings'])
        else:
            check('corrupt_image_failure_recorded',record['status']=='failed',record['warnings'])
        if record['status'] != 'failed':
            with Image.open(source) as raw:
                oriented = ImageOps.exif_transpose(raw).convert('RGB')
            valid = True
            for panel in record['panels']:
                x,y,w,h=panel['bbox']
                valid &= 0<=x<x+w<=oriented.width and 0<=y<y+h<=oriented.height
                with Image.open(out/record['page_id']/panel['crop_file']) as crop:
                    valid &= crop.size==(w,h) and crop.tobytes()==oriented.crop((x,y,x+w,y+h)).tobytes()
            check(label+'_export_coordinates',valid,'All crops equal original-oriented RGB regions; empty set is valid')

    # Skill: benchmark from second call; separate load and avoid plotting in infer-only measurements.
    with Image.open(page9) as raw:
        real_image = ImageOps.exif_transpose(raw).convert('RGB')
    start = time.perf_counter()
    detector.detect(real_image)
    evidence['benchmark']={'warmup_ms':(time.perf_counter()-start)*1000,'repetitions':5,
        'scope':'CPU preprocess + session.run + restore + NMS; excludes model load, image IO and plotting', 'samples_ms':[]}
    for _ in range(5):
        start=time.perf_counter()
        detector.detect(real_image)
        evidence['benchmark']['samples_ms'].append((time.perf_counter()-start)*1000)
    evidence['benchmark']['mean_ms']=sum(evidence['benchmark']['samples_ms'])/5
    for confidence,nms in [(.1,.45),(.25,.45),(.5,.45),(.25,.3),(.25,.6)]:
        detector.confidence,detector.nms=confidence,nms
        results=detector.detect(real_image)
        evidence['comparisons'].append(dict(confidence=confidence,nms=nms,count=len(results),
            panels=[dict(bbox=box,confidence=score) for box,score in results]))
    counts={r['confidence']:r['count'] for r in evidence['comparisons'] if r['nms']==.45}
    check('confidence_filter_behavior',counts[.1]>=counts[.25]>=counts[.5],counts)
    nms_counts={r['nms']:r['count'] for r in evidence['comparisons'] if r['confidence']==.25}
    # No assumed accuracy benefit: overlap graph can make greedy NMS non-monotonic.
    check('nms_comparison_recorded',set(nms_counts)=={.3,.45,.6},nms_counts)
    evidence['all_passed']=all(c['passed'] for c in evidence['checks'])
    write_json(destination/'evidence.json',evidence)
    write_json(source_dir/'latest_run.json',{'run':str(destination.relative_to(ROOT)), 'all_passed':evidence['all_passed']})
    print('Evidence: '+str(destination/'evidence.json'),flush=True)
    return 0 if evidence['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
