from pathlib import Path
import math
ws=Path('/nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace')
names=(ws/'list.txt').read_text().splitlines()
errors=[]
for name in names:
    n=len((ws/'sequences'/name/'groundtruth.txt').read_text().splitlines())
    base=ws/'results/rgbd/rgbd-unsupervised'/name
    for suffix in ('.txt', '_confidence.value', '_time.value'):
        p=base/(name+'_001'+suffix)
        if not p.exists():
            errors.append(str(p)+' missing')
            continue
        rows=p.read_text().splitlines()
        if len(rows)!=n:
            errors.append(str(p)+' wrong length '+str(len(rows))+' != '+str(n))
        if suffix=='_confidence.value':
            if any(not row or not math.isfinite(float(row)) for row in rows[1:]):
                errors.append(str(p)+' invalid confidence')
if errors:
    raise SystemExit('INCOMPLETE RESULTS\n'+'\n'.join(errors))
print('PASS: 50 sequences, trajectory/confidence/time lengths match ground truth.')
