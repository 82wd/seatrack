"""Resumable data preparation for the remaining official-weight evaluations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

BASE = Path('/nas/liziyan103/seatrack')
RUN = BASE / 'evaluation_runs/20261013_remaining'
VOT_URL = 'https://data.votchallenge.net/vot2022/rgbd/description.json'
RGBT_URL = 'https://hf-mirror.com/datasets/xche32/rgbt234/resolve/9ceab125c5b68e6676b7e2a8eadda49501ff37e0/rgbt234.tar.gz'
RGBT_SHA = '137f45fa9860e2ce7c5f882142703c6176ffce604cfcc1cebced3364b9cbc1ee'

def status(task, state):
    (RUN / task).mkdir(parents=True, exist_ok=True)
    (RUN / task / 'status.txt').write_text(state + '\n')
    print(task, state, flush=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def download(url, target, size=None, digest=None):
    def verify_size(path):
        if size is None or path.stat().st_size == size:
            return
        assert target.suffix == '.zip' and zipfile.is_zipfile(path), ('size', target)
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None, ('zip CRC', target)
        print('Metadata size differs; ZIP CRC passed:', target.name,
              'expected', size, 'actual', path.stat().st_size, flush=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + '.part')
        # Bound stalled transfers; retain partial bytes for an explicit retry.
        print('Downloading:', target.name, flush=True)
        subprocess.run(['curl', '--http1.1', '-fL', '--no-progress-meter', '--connect-timeout', '20',
                        '--speed-time', '120', '--speed-limit', '1024',
                        '--retry', '5', '--retry-delay', '5', '-C', '-',
                        url, '-o', str(partial)], check=True)
        if size is not None:
            verify_size(partial)
        if digest:
            assert sha(partial) == digest, ('SHA256', target)
        os.replace(partial, target)
    if size is not None:
        verify_size(target)
    if digest:
        assert sha(target) == digest, ('SHA256', target)

def safe_zip(path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            p = Path(info.filename)
            assert not p.is_absolute() and '..' not in p.parts
            assert ((info.external_attr >> 16) & 0o170000) != 0o120000
        assert z.testzip() is None, ('zip CRC', path)
        z.extractall(dest)

def prepare_rgbt():
    status('rgbt234', 'DOWNLOADING')
    archive = BASE / 'downloads/rgbt234/rgbt234.tar.gz'
    download(RGBT_URL, archive, 7665568241, RGBT_SHA)
    status('rgbt234', 'EXTRACTING')
    done = RUN / 'rgbt234/extraction.complete'
    if not done.exists():
        with tarfile.open(archive, 'r:gz') as t:
            for member in t:
                p = Path(member.name)
                assert not p.is_absolute() and '..' not in p.parts
                assert p.parts and p.parts[0] == 'rgbt234', member.name
                assert member.isfile() or member.isdir(), member.name
                t.extract(member, BASE / 'datasets')
        done.touch()
    import cv2
    import numpy as np
    root = BASE / 'datasets/rgbt234'
    seqs = sorted(p for p in root.iterdir() if p.is_dir())
    assert len(seqs) == 234, len(seqs)
    frames = 0
    for seq in seqs:
        counts = []
        for channel in ['visible', 'infrared']:
            files = sorted((seq/channel).glob('*.jpg'))
            gt = np.loadtxt(seq/(channel+'.txt'), delimiter=',', ndmin=2)
            assert gt.shape == (len(files), 4) and len(files)
            assert np.isfinite(gt).all()
            for f in (files[0], files[len(files)//2], files[-1]):
                assert cv2.imread(str(f)) is not None, f
            counts.append(len(files))
        assert counts[0] == counts[1], seq
        frames += counts[0]
    (RUN/'rgbt234/data_validation.json').write_text(json.dumps(
        dict(sequences=234, frames=frames, archive_sha256=RGBT_SHA), indent=2))
    status('rgbt234', 'DATA_READY')

def prepare_vot():
    status('vot22', 'DOWNLOADING')
    meta_path = RUN / 'vot22/description.json'
    download(VOT_URL, meta_path)
    meta = json.loads(meta_path.read_text())
    assert len(meta['sequences']) == 127
    root = BASE / 'datasets/vot2022_rgbd'
    cache = BASE / 'downloads/vot2022_rgbd'
    def prepare_sequence(seq):
        name = seq['name']
        dest = root / name
        marker = RUN / 'vot22' / (name+'.complete')
        if marker.exists():
            return name
        ann = seq['annotations']
        archive = cache/(name+'_annotations.zip')
        download(urljoin(VOT_URL, ann['url']), archive, ann['compressed'])
        safe_zip(archive, dest)
        props = dict(name=name, fps=seq['fps'], format='default')
        for channel, info in seq['channels'].items():
            archive = cache/(name+'_'+channel+'.zip')
            download(urljoin(VOT_URL, info['url']), archive, info['compressed'])
            safe_zip(archive, dest/channel)
            props['channels.'+channel] = channel+'/'+info.get('pattern', '')
            from glob import glob
            pattern = '*.jpg' if channel == 'color' else '*.png'
            assert len(glob(str(dest/channel/pattern))) == seq['length'], name
        (dest/'sequence').write_text(''.join(k+'='+str(v)+'\n' for k,v in props.items()))
        marker.touch()
        print('VOT sequence ready:', name, flush=True)
        return name
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, name in enumerate(pool.map(prepare_sequence, meta['sequences']), 1):
            print('VOT ordered progress:', i, '/127', name, flush=True)
    (root/'list.txt').write_text(''.join(s['name']+'\n' for s in meta['sequences']))
    from vot.dataset import load_dataset
    dataset = load_dataset(str(root))
    assert len(dataset) == 127
    for seq in meta['sequences']:
        assert len(dataset[seq['name']]) == seq['length']
    status('vot22', 'DATA_READY')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['rgbt234', 'vot22'])
    task = p.parse_args().task
    try:
        (prepare_rgbt if task == 'rgbt234' else prepare_vot)()
    except Exception as e:
        status(task, 'FAILED: '+repr(e))
        raise
