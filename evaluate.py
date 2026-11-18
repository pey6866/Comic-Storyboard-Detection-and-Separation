"""Evaluate manually verified xywh annotations against one run.json."""
import argparse
import json
from pathlib import Path
from panel_assistant import iou, write_json


def match(predictions, truths, threshold):
    # Maximum cardinality bipartite matching, with descending IoU tie preference.
    edges = [sorted((j for j, t in enumerate(truths) if iou(p, t) >= threshold),
                    key=lambda j: -iou(p, truths[j])) for p in predictions]
    assigned = {}

    def augment(i, visited):
        for j in edges[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or augment(assigned[j], visited):
                assigned[j] = i
                return True
        return False

    tp = sum(augment(i, set()) for i in range(len(predictions)))
    return tp, len(predictions)-tp, len(truths)-tp


def metrics(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0,
                recall=tp/(tp+fn) if tp+fn else 0, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('truth', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--output', type=Path, default=Path('evaluation/metrics.json'))
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        parser.error('IoU must be in (0,1]')
    truth = json.loads(args.truth.read_text(encoding='utf-8'))
    run = json.loads(args.run.read_text(encoding='utf-8'))
    predictions = {p['source']: p for p in run['pages']}
    rows = []
    for item in truth['pages']:
        if item.get('verified') is not True:
            raise ValueError('Unverified ground truth: ' + item['source'])
        for box in item['boxes']:
            if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                raise ValueError('Invalid ground truth box')
        prediction = predictions.get(str(Path(item['source']).resolve()))
        failed = prediction is None or prediction['status'] == 'failed'
        boxes = [] if failed else [p['bbox'] for p in prediction['panels']]
        rows.append(dict(source=item['source'], failed=failed,
                         elapsed_ms=prediction.get('elapsed_ms') if prediction else None,
                         **metrics(*match(boxes, item['boxes'], args.iou))))
    totals = [sum(row[k] for row in rows) for k in ('tp', 'fp', 'fn')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(iou=args.iou, matching='maximum-cardinality one-to-one',
               zero_denominator=0, failed_pages=sum(r['failed'] for r in rows),
               summary=metrics(*totals), pages=rows))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
