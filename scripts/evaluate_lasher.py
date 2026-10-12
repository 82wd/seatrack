#!/usr/bin/env python3
"""CPU-only LasHeR OPE analysis of existing xywh predictions (Python >=3.8).

See lasher_metrics.md for reference source, protocol, and verification limits.
This is a Python implementation of a public MATLAB toolkit, not an official
evaluator. It never modifies predictions or ground truth.
"""
import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REFERENCE = ('https://github.com/xuboyue1999/RGBT-Tracking/tree/'
             '3613a3cbc9d0626a54c7755a549241f658ef6d46/LasHeR_Toolkit')
ERROR_THRESHOLDS = np.arange(51, dtype=float)
NORM_THRESHOLDS = ERROR_THRESHOLDS / 100
IOU_THRESHOLDS = np.arange(21, dtype=float) / 20


def load_boxes(path):
    raw = path.read_bytes()
    # Both dataset CSV and np.savetxt whitespace output are accepted.
    boxes = np.loadtxt(io.StringIO(raw.decode('utf-8-sig').replace(',', ' ')),
                       ndmin=2)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
        raise ValueError('{}: expected nonempty N x 4 xywh'.format(path))
    if not np.isfinite(boxes).all():
        raise ValueError('{}: contains NaN/Inf'.format(path))
    return boxes, hashlib.sha256(raw).hexdigest()


def sequence_curves(prediction, gt):
    if prediction.shape != gt.shape or not len(gt):
        raise ValueError('Prediction and annotation shapes must match (nonempty N x 4)')
    if not np.isfinite(prediction).all() or not np.isfinite(gt).all():
        raise ValueError('Nonfinite input requires inspection, not silent repair')
    pred = prediction.copy()
    repaired = 0
    # The reference carries forward invalid boxes before replacing frame one.
    for i in range(1, len(pred)):
        if np.any(pred[i, 2:] <= 0):
            pred[i] = pred[i - 1]
            repaired += 1
    pred[0] = gt[0]
    center = pred[:, :2] + (pred[:, 2:] - 1) / 2
    gt_center = gt[:, :2] + (gt[:, 2:] - 1) / 2
    distance = np.sqrt(np.sum((center - gt_center) ** 2, axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        # Preserve the reference's divide-centers-then-subtract order.
        normalized = np.sqrt(np.sum(
            (center / gt[:, 2:] - gt_center / gt[:, 2:]) ** 2, axis=1))
    valid = np.all(gt > 0, axis=1)
    overlap = np.full(len(gt), -1.0)
    a, b = pred[valid], gt[valid]
    edge = np.maximum(0, np.minimum(a[:, :2] + a[:, 2:] - 1,
                                   b[:, :2] + b[:, 2:] - 1)
                      - np.maximum(a[:, :2], b[:, :2]) + 1)
    intersection = edge[:, 0] * edge[:, 1]
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - intersection
    with np.errstate(divide='ignore', invalid='ignore'):
        overlap[valid] = intersection / union
    # Deliberately preserve the reference's unusual invalid-GT rule:
    # -1 is a success for distance thresholds, a failure for IoU thresholds.
    distance[~valid] = -1
    normalized[~valid] = -1
    denominator = len(gt) + np.finfo(float).eps
    curves = {
        'precision': (distance[:, None] <= ERROR_THRESHOLDS).sum(0) / denominator,
        'normalized_precision': (normalized[:, None] <= NORM_THRESHOLDS).sum(0) / denominator,
        'success': (overlap[:, None] > IOU_THRESHOLDS).sum(0) / denominator,
    }
    return curves, {'frames': len(gt), 'invalid_gt_frames': int((~valid).sum()),
                    'carried_forward_predictions': repaired}


def summarize(curves_list):
    curves, excluded = {}, {}
    for key in ('precision', 'normalized_precision', 'success'):
        matrix = np.stack([c[key] for c in curves_list])
        keep = matrix.sum(axis=1) > np.finfo(float).eps
        if not keep.any():
            raise ValueError('No nonzero curves for {}'.format(key))
        curves[key] = matrix[keep].mean(axis=0)
        excluded[key] = np.flatnonzero(~keep).tolist()
    scores = {'PR': float(curves['precision'][20]),
              'NPR': float(curves['normalized_precision'][20]),
              'SR': float(curves['success'].mean())}
    return scores, curves, excluded


def evaluate(data_root, results, expected_sequences=245, annotation_root=None):
    seqs = sorted(p for p in data_root.iterdir() if p.is_dir())
    names = {p.name for p in seqs}
    found = {p.stem for p in results.glob('*.txt')}
    if len(names) != expected_sequences or names != found:
        raise ValueError('Expected {} sequences, found {}; missing={}; extra={}'.format(
            expected_sequences, len(names), sorted(names-found), sorted(found-names)))
    all_curves, records = [], []
    for seq in seqs:
        gt_path = annotation_root / (seq.name + '.txt') if annotation_root else seq / 'visible.txt'
        pred_path = results / (seq.name + '.txt')
        gt, gt_hash = load_boxes(gt_path)
        pred, pred_hash = load_boxes(pred_path)
        if len(pred) != len(gt):
            raise ValueError('{}: prediction/GT lengths differ'.format(seq.name))
        curves, counts = sequence_curves(pred, gt)
        all_curves.append(curves)
        records.append(dict(name=seq.name, **counts,
                            annotation_sha256=gt_hash, prediction_sha256=pred_hash,
                            PR=float(curves['precision'][20]),
                            NPR=float(curves['normalized_precision'][20]),
                            SR=float(curves['success'].mean()),
                            curves={k: v.tolist() for k, v in curves.items()}))
    scores, curves, excluded = summarize(all_curves)
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reference': REFERENCE,
        'verification': 'Source-reviewed Python port; not executed against MATLAB. '
                        'Dataset visible.txt vs toolkit annos equivalence is not verified.'
                        if annotation_root is None else
                        'Source-reviewed Python port; not executed against MATLAB. '
                        'Uses supplied reference annotations.',
        'data_root': str(data_root.resolve()), 'results': str(results.resolve()),
        'annotation_source': str(annotation_root.resolve()) if annotation_root else 'dataset visible.txt',
        'sequences': len(records), 'frames': sum(r['frames'] for r in records),
        'scores': scores, 'percent': {k: v*100 for k, v in scores.items()},
        'thresholds': {'precision': ERROR_THRESHOLDS.tolist(),
                       'normalized_precision': NORM_THRESHOLDS.tolist(),
                       'success': IOU_THRESHOLDS.tolist()},
        'average_curves': {k: v.tolist() for k, v in curves.items()},
        'excluded_zero_curves': {k: [seqs[i].name for i in ids] for k, ids in excluded.items()},
        'per_sequence': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annotation-root', type=Path,
                        help='Optional reference toolkit annos directory (<sequence>.txt)')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new report filename')
    report = evaluate(args.data_root, args.results, annotation_root=args.annotation_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print('Sequences: {}; frames: {}'.format(report['sequences'], report['frames']))
    print('PR / NPR / SR: {:.4f} / {:.4f} / {:.4f} (%)'.format(
        report['percent']['PR'], report['percent']['NPR'], report['percent']['SR']))
    print('Invalid GT frames: {}; carried-forward predictions: {}'.format(
        sum(r['invalid_gt_frames'] for r in report['per_sequence']),
        sum(r['carried_forward_predictions'] for r in report['per_sequence'])))
    print('Report: {}'.format(args.output))
    print('Verification: {}'.format(report['verification']))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        print('ERROR: {}'.format(exc), file=sys.stderr)
        sys.exit(1)
