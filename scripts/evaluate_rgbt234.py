#!/usr/bin/env python3
"""RGBT234 NumPy evaluation; protocol port, not official MATLAB validation.

Reference: https://github.com/opacity-black/RGBT_toolkit
src/rgbt/{metrics/metrics.py,dataset/basedataset.py,utils.py}, read 2026-10-13.
Primary result follows reference prediction rounding. Raw-float diagnostic is
reported separately. Equal sequence weighting; no frame removal or replacement.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def read_boxes(path):
    data = path.read_bytes()
    # StringIO defaults do not split CR-only lines; normalize in memory only.
    content = data.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    try:
        boxes = np.loadtxt(io.StringIO(content.replace(',', ' ')),
                           dtype=np.float32, ndmin=2)
    except ValueError as exc:
        raise ValueError('Cannot parse {}: {}'.format(path, exc)) from exc
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
        raise ValueError('Invalid shape: {} {}'.format(path, boxes.shape))
    if not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError('Nonfinite or nonpositive box size: {}'.format(path))
    return boxes, hashlib.sha256(data).hexdigest()


def curves(pred, visible, infrared):
    # Calculate in float64 after the reference's float32 loading/rounding.
    pred, visible, infrared = [x.astype(np.float64) for x in (pred, visible, infrared)]
    distances, overlaps = [], []
    for gt in (visible, infrared):
        delta = pred[:, :2] + (pred[:, 2:] - 1) / 2 - (gt[:, :2] + (gt[:, 2:] - 1) / 2)
        distances.append(np.sqrt(np.sum(delta ** 2, axis=1)))
        wh = np.maximum(0, np.minimum(pred[:, :2] + pred[:, 2:],
                                      gt[:, :2] + gt[:, 2:]) - np.maximum(pred[:, :2], gt[:, :2]))
        intersection = wh.prod(axis=1)
        union = pred[:, 2:].prod(axis=1) + gt[:, 2:].prod(axis=1) - intersection
        overlaps.append(intersection / union)
    distance = np.minimum(*distances)
    overlap = np.maximum(*overlaps)
    precision = (distance[:, None] <= np.arange(51)).mean(axis=0)
    success = (overlap[:, None] > np.linspace(0, 1, 21)).mean(axis=0)
    return precision, success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seqs = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    if len(seqs) != 234:
        raise ValueError('Expected 234 sequences, got {}'.format(len(seqs)))
    expected = {s.name + '.txt' for s in seqs}
    actual = {p.name for p in args.results.glob('*.txt')}
    if expected != actual:
        raise ValueError('Missing: {}; extra: {}'.format(sorted(expected-actual), sorted(actual-expected)))
    report = {'dataset': str(args.dataset), 'results': str(args.results),
              'reference': 'https://github.com/opacity-black/RGBT_toolkit',
              'qualification': 'NumPy protocol port; not cross-validated with official MATLAB',
              'protocol': 'float32 input; nearest-even prediction rounding; per-frame min center error / max IoU over modalities; <=20px; IoU > thresholds 0:0.05:1; equal sequence mean; all frames retained',
              'sequences': {}, 'frames': 0}
    all_curves = {'rounded_reference_protocol': [], 'raw_float_diagnostic': []}
    for index, seq in enumerate(seqs, 1):
        pred, phash = read_boxes(args.results / (seq.name + '.txt'))
        vis, vhash = read_boxes(seq / 'visible.txt')
        ir, ihash = read_boxes(seq / 'infrared.txt')
        if pred.shape != vis.shape or pred.shape != ir.shape:
            raise ValueError('Shape mismatch: {}'.format(seq.name))
        rounded = pred.round(0)
        if (rounded[:, 2:] <= 0).any():
            raise ValueError('Rounding creates nonpositive size: {}'.format(seq.name))
        item = {'frames': len(pred), 'sha256': {'prediction': phash, 'visible': vhash, 'infrared': ihash}}
        for name, boxes in [('rounded_reference_protocol', rounded), ('raw_float_diagnostic', pred)]:
            pr, sr = curves(boxes, vis, ir)
            all_curves[name].append((pr, sr))
            item[name] = {'MPR_percent': float(pr[20] * 100), 'MSR_percent': float(sr.mean() * 100)}
        report['sequences'][seq.name] = item
        report['frames'] += len(pred)
        if index % 40 == 0 or index == len(seqs):
            print('Evaluated {}/{}'.format(index, len(seqs)), flush=True)
    report['sequence_count'] = len(seqs)
    for name, values in all_curves.items():
        pr = np.mean([v[0] for v in values], axis=0)
        sr = np.mean([v[1] for v in values], axis=0)
        report[name] = {'MPR_percent': float(pr[20]*100), 'MSR_percent': float(sr.mean()*100),
                        'precision_curve': pr.tolist(), 'success_curve': sr.tolist()}
        print('{}: MPR={:.6f}% MSR={:.6f}%'.format(name, pr[20]*100, sr.mean()*100))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print('Sequences: {}; frames: {}'.format(len(seqs), report['frames']))
    print('Report:', args.output)


if __name__ == '__main__':
    main()
