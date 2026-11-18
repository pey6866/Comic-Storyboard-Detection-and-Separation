"""Recompute archived geometric metrics without model weights or source images."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import match, metrics


def main():
    folder = ROOT/'evaluation/published/20261116_112754'
    load = lambda name: json.loads((folder/name).read_text(encoding='utf-8-sig'))
    truth = load('test_truth_snapshot.json')['pages']
    comparison = load('comparison.json')
    checks = 0
    for method in ('inkwell_default', 'inkwell_selected', 'opencv_baseline'):
        predictions = {p['source']: p for p in load(method+'/run.json')['pages']}
        for expected in comparison[method]['metrics']:
            rows = []
            for page in truth:
                prediction = predictions.get(page['source'])
                failed = prediction is None or prediction['status'] == 'failed'
                boxes = [] if failed else [p['bbox'] for p in prediction['panels']]
                row = metrics(*match(boxes, page['boxes'], expected['iou']))
                saved = next(r for r in expected['pages'] if r['source'] == page['source'])
                for key in row:
                    if abs(row[key]-saved[key]) > 1e-12:
                        raise ValueError(f'{method}: per-page {key} differs')
                rows.append(row)
            actual = metrics(*(sum(r[k] for r in rows) for k in ('tp','fp','fn')))
            if actual != expected['summary']:
                raise ValueError(f'{method}: summary differs at {expected["iou"]}')
            checks += 1
            macro = sum(row['f1'] for row in rows)/len(rows)
            print(f'PASS {method} IoU={expected["iou"]:.2f} '
                  f'TP={actual["tp"]} FP={actual["fp"]} FN={actual["fn"]} '
                  f'micro_F1={actual["f1"]:.6f} macro_F1={macro:.6f}')
    grid = load('development_grid.json')
    for row in grid:
        aggregate = metrics(*(sum(p[k] for p in row['pages']) for k in ('tp','fp','fn')))
        if aggregate != row['summary']:
            raise ValueError('Development grid summary differs')
    best = max(grid, key=lambda r: (r['summary']['f1'], r['summary']['precision'],
                                    -abs(r['confidence']-.25)-abs(r['nms']-.45)))
    frozen = load('selected_parameters.json')
    if any(best[k] != frozen[k] for k in ('confidence','nms')):
        raise ValueError('Selected parameters differ')
    print(f'PASS development selection: confidence={best["confidence"]}, NMS={best["nms"]}')
    evidence = json.loads((ROOT/'docs/skill_reuse/evidence_public.json').read_text(encoding='utf-8'))
    if len(evidence['checks']) != 13 or not all(c['passed'] for c in evidence['checks']):
        raise ValueError('Historical skill evidence is incomplete')
    print(f'PASS {checks} metric comparisons, 15 development rows, 13 archived skill checks')
    print('Statistics recomputed from saved geometry; no inference, image review or timing repeated.')


if __name__ == '__main__':
    main()
