"""Audit manual truth, select on development set, freeze, evaluate held-out pages."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import hashlib
import json
import math
import statistics
import time
from datetime import datetime
from PIL import Image, ImageOps, ImageDraw
from panel_assistant import ROOT, Inkwell, Baseline, process, write_json, iou, draw_panels
from evaluate import match, metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(records, truth, threshold=.5):
    indexed={r['source']:r for r in records}
    rows=[]
    for page in truth:
        record=indexed.get(page['source'])
        failed=not record or record['status']=='failed'
        boxes=[] if failed else [p['bbox'] for p in record['panels']]
        rows.append(dict(source=page['source'],failed=failed,
            inference_ms=record.get('inference_ms') if record else None,
            elapsed_ms=record.get('elapsed_ms') if record else None,
            **metrics(*match(boxes,page['boxes'],threshold))))
    return dict(iou=threshold,matching='maximum-cardinality one-to-one',zero_denominator=0,
        summary=metrics(*(sum(r[k] for r in rows) for k in ('tp','fp','fn'))),
        failed_pages=sum(r['failed'] for r in rows),pages=rows)


def main():
    if any(not (ROOT/'evaluation'/(s+'_truth.json')).is_file() for s in ('dev','test')):
        raise SystemExit('No local verified dev/test annotations. See docs/REPRODUCE.md; use scripts/check_published_metrics.py for the published snapshot.')
    out=ROOT/'evaluation'/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    audit=dict(errors=[],warnings=[],splits={},annotation_quality='User estimates above 95%; subjective, not measured or calibrated',
               matching='IoU>=0.5, maximum-cardinality one-to-one; micro averages; zero denominator=0')
    truth={}
    all_sources=set()
    all_hashes=set()
    for split in ('dev','test'):
        path=ROOT/'evaluation'/(split+'_truth.json')
        raw=json.loads(path.read_text(encoding='utf-8-sig'))
        write_json(out/(split+'_truth_snapshot.json'),raw)
        truth[split]=sorted(raw['pages'],key=lambda p:p['source'])
        for page in truth[split]:
            source=Path(page['source'])
            if page.get('verified') is not True or page.get('split')!=split:
                audit['errors'].append('Invalid verification/split: '+str(source))
            if not source.exists():
                audit['errors'].append('Missing source: '+str(source));continue
            digest=sha(source)
            if digest!=page.get('source_sha256'):
                audit['errors'].append('Image hash mismatch: '+str(source))
            if str(source) in all_sources or digest in all_hashes:
                audit['errors'].append('Duplicate image across/within splits: '+str(source))
            all_sources.add(str(source));all_hashes.add(digest)
            with Image.open(source) as raw_image:
                image=ImageOps.exif_transpose(raw_image)
                w,h=image.size
            if page.get('image_size')!=[w,h]:
                audit['errors'].append('Image size mismatch: '+str(source))
            if not page['boxes']:
                audit['warnings'].append('Confirmed empty truth: '+str(source))
            for b in page['boxes']:
                if len(b)!=4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in b):
                    audit['errors'].append('Invalid box: '+str(source));continue
                x,y,bw,bh=b
                if x<0 or y<0 or bw<=0 or bh<=0 or x+bw>w or y+bh>h:
                    audit['errors'].append('Out-of-bounds box: '+str(source))
            for i,a in enumerate(page['boxes']):
                for b in page['boxes'][i+1:]:
                    if iou(a,b)>.9:
                        audit['warnings'].append('Highly overlapping truth boxes (not removed): '+str(source))
        audit['splits'][split]=dict(pages=len(truth[split]),boxes=sum(len(p['boxes']) for p in truth[split]),
            files=[Path(p['source']).name for p in truth[split]],sha256=sha(path))
    if not truth['dev'] or not truth['test']:
        audit['errors'].append('Both splits must be nonempty')
    audit['warnings'].append('Same comic issue; not an independent multi-work benchmark; pretraining overlap unknown')
    write_json(out/'audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    if audit['errors']:
        raise ValueError('Audit failed; no parameter selection or test prediction executed')
    # Freeze experiment protocol before any held-out prediction.
    grid=[dict(confidence=c,nms=n) for c in (.1,.25,.4,.5,.65) for n in (.3,.45,.6)]
    write_json(out/'protocol.json',dict(grid=grid,selection='max dev micro-F1@.5, then precision, then proximity to defaults .25/.45',
        test_iou_thresholds=[.3,.5,.75],default=dict(confidence=.25,nms=.45),imgsz=640,
        annotation_policy='Manual rough rectangles, no automatic relabeling; 95% is subjective',
        code_sha256={p:sha(ROOT/p) for p in ('panel_assistant.py','evaluate.py','scripts/run_evaluation.py')}))
    detector=Inkwell(ROOT/'models/best.onnx',.25,.45)
    images={}
    for p in truth['dev']:
        with Image.open(p['source']) as raw:
            images[p['source']]=ImageOps.exif_transpose(raw).convert('RGB')
    grid_results=[]
    for params in grid:
        detector.confidence,detector.nms=params['confidence'],params['nms']
        records=[]
        for source,image in images.items():
            result=detector.detect(image)
            records.append(dict(source=source,status='success',panels=[dict(bbox=b,confidence=s) for b,s in result]))
        evaluated=score(records,truth['dev'])
        grid_results.append(dict(**params,**evaluated))
        print('DEV',params,evaluated['summary'],flush=True)
    write_json(out/'development_grid.json',grid_results)
    selected=max(grid_results,key=lambda r:(r['summary']['f1'],r['summary']['precision'],
        -abs(r['confidence']-.25)-abs(r['nms']-.45)))
    params={k:selected[k] for k in ('confidence','nms')}
    write_json(out/'selected_parameters.json',dict(**params,review_threshold=.5,imgsz=640,
        selected_at=datetime.now().astimezone().isoformat(),dev_metrics=selected['summary'],model=detector.metadata,
        note='Frozen before test inference; test results must not be used to reselect'))
    print('FROZEN',params,flush=True)
    all_methods={}
    methods=[('inkwell_default',Inkwell(ROOT/'models/best.onnx',.25,.45),dict(confidence=.25,nms=.45)),
             ('inkwell_selected',detector,params),('opencv_baseline',Baseline(),dict(confidence=.25,nms=.45))]
    detector.confidence,detector.nms=params['confidence'],params['nms']
    for name,model,config in methods:
        folder=out/name
        folder.mkdir()
        # Consistent warmup on development page, not held-out page.
        model.detect(next(iter(images.values())))
        records=[process(Path(p['source']),folder,model,name,dict(**config,review_threshold=.5)) for p in truth['test']]
        write_json(folder/'run.json',dict(pages=records,model=model.metadata,
            timing='warm model; inference_ms includes preprocess/run/NMS; elapsed_ms includes IO and all image exports; excludes model loading'))
        evaluations=[score(records,truth['test'],t) for t in (.3,.5,.75)]
        for evaluation in evaluations:
            write_json(folder/f"metrics_iou_{evaluation['iou']:.2f}.json",evaluation)
        times=[r['inference_ms'] for r in records if 'inference_ms' in r]
        elapsed=[r['elapsed_ms'] for r in records]
        all_methods[name]=dict(parameters=config,metrics=evaluations,timing=dict(mean_inference_ms=statistics.mean(times) if times else None,
            median_inference_ms=statistics.median(times) if times else None,mean_elapsed_ms=statistics.mean(elapsed)),records=records)
        print('TEST',name,evaluations[1]['summary'],flush=True)
    write_json(out/'comparison.json',all_methods)
    with (out/'per_page.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['method','source','tp','fp','fn','precision','recall','f1','inference_ms','elapsed_ms','failed']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for name,result in all_methods.items():
            for row in result['metrics'][1]['pages']:
                writer.writerow(dict(method=name,**row))
    # Local visual review of truth and predictions; do not edit ground truth based on test outcomes.
    selected_rows=all_methods['inkwell_selected']['metrics'][1]['pages']
    ranked=sorted(selected_rows,key=lambda r:-(r['fp']+r['fn']))[:3]
    examples=[]
    for rank,row in enumerate(ranked,1):
        page=next(p for p in truth['test'] if p['source']==row['source'])
        with Image.open(page['source']) as raw:
            original=ImageOps.exif_transpose(raw).convert('RGB')
        canvas=Image.new('RGB',(1200,670),'#eeeeee')
        draw=ImageDraw.Draw(canvas)
        overlays=[('Manual truth',draw_panels(original,[dict(id=i+1,bbox=b) for i,b in enumerate(page['boxes'])],True))]
        for method in ('inkwell_selected','opencv_baseline'):
            record=next(r for r in all_methods[method]['records'] if r['source']==row['source'])
            overlays.append((method,draw_panels(original,record['panels'],True)))
        for col,(title,image) in enumerate(overlays):
            image.thumbnail((390,620));canvas.paste(image,(col*400,35));draw.text((col*400+8,10),title,fill='black')
        filename=f'case_{rank}.jpg';canvas.save(out/filename)
        examples.append(dict(file=filename,source=page['source'],metrics=row))
    write_json(out/'failure_cases.json',examples)
    lines=['# 模型与基线评估结果','',f'实验目录：{out.name}。',
        f"人工核验开发集 {len(truth['dev'])} 页 / {audit['splits']['dev']['boxes']} 框；保留测试集 {len(truth['test'])} 页 / {audit['splits']['test']['boxes']} 框。",
        '',f"开发集选定 confidence={params['confidence']}，NMS={params['nms']}，输入640，CPUExecutionProvider。按开发集F1、Precision、接近默认参数顺序选择；随后冻结并评估测试。",
        '', '| 方法 | Precision | Recall | F1 | TP | FP | FN | 平均推理 ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,result in all_methods.items():
        m=result['metrics'][1]['summary']
        lines.append(f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} | {result['timing']['mean_inference_ms']:.2f} |")
    lines+=['','主指标采用 IoU≥0.5、最大基数一对一匹配、微平均，零分母记0。失败/缺失预测按空预测处理并单列失败页，不排除难例。',
        '', '| 方法 | F1 @ IoU .30 | F1 @ IoU .50 | F1 @ IoU .75 |','|---|---:|---:|---:|']
    for name,result in all_methods.items():
        lines.append('| '+name+' | '+' | '.join(f"{m['summary']['f1']:.4f}" for m in result['metrics'])+' |')
    lines+=['','手工拖框存在边界误差。用户认为标注质量在95%以上，这是主观估计，不是经独立复核得到的准确率、置信区间或模型阈值；评估按原始人工框直接匹配，不按预测自动修正。IoU敏感性仅说明结果随匹配严格程度变化，不能证明标注误差大小。',
        '', '同一册漫画，小样本、单人标注；页面缩略图曾用于版式检查，跨作品泛化与预训练数据重叠未知。没有按普通/异形页预先建立类别标签，因此不虚构分组指标。',
        '', '计时：各方法使用开发页预热，测试每页单次运行；推理含预处理/模型/后处理，不含读图及导出；端到端另见per_page.csv。CPU负载、文件缓存及方法执行顺序会影响速度，不做严格性能优劣结论。',
        '', '逐页CSV、全量预测/裁剪、IoU敏感性、开发参数网格、冻结配置、数据快照与审计均在本目录。case_1至case_3按选定模型的FP+FN排序，展示人工框/模型/基线；即使误差为0也只是展示样例。']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(ROOT/'evaluation/latest_evaluation.json',dict(directory=str(out.relative_to(ROOT)),parameters=params))
    print('OUTPUT',out,flush=True)


if __name__=='__main__':
    main()
