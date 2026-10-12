from pathlib import Path
import shutil, re, hashlib, json
BASE=Path('/nas/liziyan103/seatrack')
RUN=BASE/'training_runs/20261011_remaining'
SRC=BASE/'SEATrack/SEATrack-main'
CODE=RUN/'code'
assert not (RUN/'launch_ready').exists()
CODE.mkdir(exist_ok=True)
for name in ('lib','experiments','tracking'):
    if not (CODE/name).exists():
        shutil.copytree(SRC/name,CODE/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
p=CODE/'lib/train/admin/local.py'
s=p.read_text()
paths={'workspace_dir':str(CODE),'tensorboard_dir':str(RUN/'tensorboard'),'lasher_dir':str(BASE/'datasets/lasher/trainingset'),'visevent_dir':str(BASE/'datasets/visevent/train_subset')}
for k,v in paths.items():
    s,n=re.subn(r'(self\.'+k+r'\s*=\s*)[^\n]+',lambda m:m[1]+repr(v),s)
    assert n==1,k
p.write_text(s)
for task in ('rgbt','rgbe'):
    p=CODE/f'experiments/seatrack/{task}.yaml';s=p.read_text()
    s=re.sub(r'PRETRAIN_FILE:.*', 'PRETRAIN_FILE: "'+str(SRC/'pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar')+'"',s)
    s=re.sub(r'SAVE_EPOCH_INTERVAL: \d+','SAVE_EPOCH_INTERVAL: 1',s)
    if task=='rgbe': s=s.replace('TEST:\n  EPOCH: 60','TEST:\n  EPOCH: 45')
    p.write_text(s)
p=CODE/'lib/train/train_script.py'; s=p.read_text();s=s.replace('fail_safe=True','fail_safe=False');p.write_text(s)
# Exact numeric frame IDs avoid suffix matches such as 5 matching 15.jpg.
p=CODE/'lib/train/dataset/lasher.py';s=p.read_text()
start=s.index('    def _get_frame_path(');end=s.index('    def _get_frame(',start)
s=s[:start]+'''    def _get_frame_path(self, seq_path, frame_id):
        import re
        if not hasattr(self, '_frame_cache'):
            self._frame_cache = {}
        if seq_path not in self._frame_cache:
            pairs = []
            for channel in ('visible', 'infrared'):
                mapping = {}
                for path in glob.glob(os.path.join(seq_path, channel, '*.jpg')):
                    match = re.search(r'(\\d+)\\.jpg$', os.path.basename(path))
                    if match is None:
                        raise ValueError('Unrecognized frame name: ' + path)
                    key = int(match.group(1))
                    if key in mapping:
                        raise ValueError('Duplicate frame ID: ' + path)
                    mapping[key] = path
                pairs.append(mapping)
            if set(pairs[0]) != set(pairs[1]):
                raise ValueError('Unpaired modalities: ' + seq_path)
            self._frame_cache[seq_path] = pairs
        pairs = self._frame_cache[seq_path]
        return pairs[0][frame_id], pairs[1][frame_id]

'''+s[end:];p.write_text(s)
manifest={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.rglob('*') if p.is_file()}
(RUN/'code_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Configured isolated code:',CODE)
