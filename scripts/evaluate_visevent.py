"""VisEvent reference metrics port for already validated SEATrack trajectories.
Uses official annos.zip. This is not a MATLAB cross-validated implementation.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import zipfile
import numpy as np

SPECIAL = "dvSave-2021_02_06_09_23_50_person1"
SOURCE = "https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark"
EPS = np.finfo(float).eps

def curves(pred, gt, absent):
    # Official calc_seq_err_robust truncates excess results and resets frame 1.
    n = len(gt)
    assert gt.shape == (n, 4) and n > 0
    assert pred.ndim == 2 and pred.shape[1] == 4 and len(pred) >= n
    assert absent.shape == (n,) and np.isin(absent, [0, 1]).all()
    assert np.isfinite(pred).all() and (pred[:, 2:] > 0).all()
    assert np.isfinite(gt).all()
    pred = pred[:n].copy()
    pred[0] = gt[0]
    keep = absent != 1
    p, g = pred[keep], gt[keep]
    valid = (g > 0).all(axis=1)
    error = np.sqrt(np.sum(((p[:, :2] + (p[:, 2:] - 1)/2) -
                            (g[:, :2] + (g[:, 2:] - 1)/2))**2, axis=1))
    # Preserve the official invalid-GT handling, including error=-1.
    error[~valid] = -1
    overlap = np.full(len(g), -1.0)
    a, b = p[valid], g[valid]
    size = np.maximum(0, np.minimum(a[:, :2]+a[:, 2:]-1,
                                   b[:, :2]+b[:, 2:]-1)
                      - np.maximum(a[:, :2], b[:, :2])+1)
    inter = size.prod(axis=1)
    overlap[valid] = inter / (a[:, 2:].prod(axis=1)+b[:, 2:].prod(axis=1)-inter)
    # Official denominator includes absent frames.
    precision = (error[:, None] <= np.arange(51)).sum(axis=0)/(n+EPS)
    success = (overlap[:, None] > np.arange(21)/20).sum(axis=0)/(n+EPS)
    return precision, success

def aggregate(rows):
    p = np.array([r["precision_curve"] for r in rows])
    s = np.array([r["success_curve"] for r in rows])
    # plot_draw_save drops all-zero curves separately for each metric.
    p = p[p.sum(axis=1) > EPS]
    s = s[s.sum(axis=1) > EPS]
    assert len(p) and len(s)
    return dict(PR20_percent=float(p.mean(axis=0)[20]*100),
                SR_AUC_percent=float(s.mean(axis=0).mean()*100),
                sequences=len(rows), precision_sequences=len(p),
                success_sequences=len(s))

def main():
    base = Path("/nas/liziyan103/seatrack")
    tool = base/"tools/visevent_benchmark"
    result = base/"SEATrack/RGBE_workspace/official_rgbe_full_20261012_035143/VisEvent"
    names = [s.strip() for s in (tool/"VisEvent_testing_subset.txt").read_text().splitlines() if s.strip()]
    assert len(names) == len(set(names)) == 320
    rows = []
    with zipfile.ZipFile(tool/"annos.zip") as z:
        def read(path):
            return np.loadtxt(io.StringIO(z.read(path).decode("utf-8-sig").replace(",", " ")), ndmin=2)
        for name in names:
            gt = read("gt_rect/"+name+".txt")
            absent = read("absent/"+name+".txt").reshape(-1)
            path = result/(name+".txt")
            pred = np.loadtxt(path, delimiter=",", ndmin=2)
            excess = len(pred)-len(gt)
            assert excess == (26 if name == SPECIAL else 0), (name, excess)
            p, s = curves(pred, gt, absent)
            rows.append(dict(sequence=name, prediction_frames=len(pred),
                             official_frames=len(gt), truncated_tail_frames=excess,
                             result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             precision_curve=p.tolist(), success_curve=s.tolist()))
    full = aggregate(rows)
    diagnostic = aggregate([r for r in rows if r["sequence"] != SPECIAL])
    special = next(r for r in rows if r["sequence"] == SPECIAL)
    report = dict(protocol="Python port of official metric rules; not MATLAB cross-validated",
                  source=SOURCE, result_directory=str(result),
                  annotation_sha256=hashlib.sha256((tool/"annos.zip").read_bytes()).hexdigest(),
                  annotation_mismatch_sequence=SPECIAL,
                  alignment_status="Unresolved for special sequence; official prefix truncation applied",
                  official_rules_320=full,
                  diagnostic_319_excluding_mismatch=diagnostic, sequences=rows)
    out = result.parent/("metrics_reference_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".json")
    with out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("320 sequences / official rules:", full)
    print("319 sequences / diagnostic ONLY:", diagnostic)
    print("Special sequence PR20/SR (%):",
          special["precision_curve"][20]*100,
          float(np.mean(special["success_curve"]))*100)
    print("Report:", out)

if __name__ == "__main__":
    main()
