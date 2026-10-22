"""Run VOT2022 RGB-D with the explicit self-trained epoch-25 checkpoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


BASE = Path('/nas/liziyan103/seatrack')
TRACKER = 'rgbd_self_ep0025'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def keep_text(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError('Existing file differs; refusing to replace: ' + str(path))
    else:
        with path.open('x') as stream:
            stream.write(text)


def prepare(base):
    project = base / 'SEATrack'
    data = base / 'datasets/vot2022_rgbd'
    checkpoint = project / 'SEATrack-main/models_rgbd_self/checkpoints/rgbd/SEATrack_ep0025.pth.tar'
    root = base / 'evaluation_runs/self_vot22_ep0025'
    names = [line.strip() for line in (data / 'list.txt').read_text().splitlines() if line.strip()]
    if len(names) != 127 or len(set(names)) != 127:
        raise ValueError('Expected 127 unique VOT2022 sequence names')
    for name in names:
        if name in ('.', '..') or Path(name).name != name:
            raise ValueError('Invalid sequence name: ' + name)
        for filename in ('sequence', 'groundtruth.txt'):
            if not (data / name / filename).is_file():
                raise FileNotFoundError(data / name / filename)
    for filename in ('lib/test/vot/seatrack_class.py', 'experiments/seatrack/rgbd.yaml'):
        if not (project / filename).is_file():
            raise FileNotFoundError(project / filename)
    manifest = dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
                    project=str(project), data=str(data), sequences=names,
                    stack='vot2022/rgbd', tracker=TRACKER)
    record = root / 'checkpoint_used.json'
    if not record.exists() and any(root.glob('*/results/**/*')):
        raise ValueError('Existing results have no checkpoint manifest')
    keep_text(record, json.dumps(manifest, indent=2) + '\n')

    # VOT imports this standalone module in each tracker child process.
    # Override parameters before run_vot_exp constructs the actual tracker.
    bridge = '''import os
import sys
sys.path.insert(0, {project!r})
os.chdir({project!r})
import lib.test.parameter.seatrack as parameter_module
original_parameters = parameter_module.parameters
original_env_settings = parameter_module.env_settings

def local_environment():
    settings = original_env_settings()
    settings.prj_dir = {project!r}
    return settings

parameter_module.env_settings = local_environment

def self_parameters(*args, **kwargs):
    params = original_parameters(*args, **kwargs)
    params.checkpoint = {checkpoint!r}
    params.cfg.MODEL.HMOE_RANK = 4
    params.cfg.MODEL.AMGLORA_RANK = 8
    print('Actual checkpoint:', params.checkpoint, file=sys.stderr, flush=True)
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'), file=sys.stderr, flush=True)
    return params

parameter_module.parameters = self_parameters
from lib.test.vot.seatrack_class import run_vot_exp
run_vot_exp('seatrack', 'rgbd', vis=False, out_conf=True, channel_type='rgbd')
'''.format(project=str(project), checkpoint=str(checkpoint))
    keep_text(root / 'self_vot22_tracker.py', bridge)
    smoke_name = 'adapter01_indoor_1' if 'adapter01_indoor_1' in names else names[0]
    for stage, selected in (('smoke', [smoke_name]), ('full', names)):
        workspace = root / stage
        sequences = workspace / 'sequences'
        sequences.mkdir(parents=True, exist_ok=True)
        keep_text(workspace / 'config.yaml', 'registry:\n- ./trackers.ini\nstack: vot2022/rgbd\n')
        keep_text(workspace / 'trackers.ini',
                  '[{}]\nlabel = {}\nprotocol = traxpython\ncommand = self_vot22_tracker\npaths = {}\n'.format(
                      TRACKER, TRACKER, root))
        keep_text(sequences / 'list.txt', '\n'.join(selected) + '\n')
        for name in selected:
            link = sequences / name
            target = data / name
            if link.is_symlink() or link.exists():
                if not link.is_symlink() or link.resolve() != target.resolve():
                    raise ValueError('Unexpected existing sequence path: ' + str(link))
            else:
                link.symlink_to(target, target_is_directory=True)
    print('Actual checkpoint:', checkpoint, flush=True)
    print('SHA256:', manifest['sha256'], flush=True)
    print('Smoke sequence:', smoke_name, flush=True)
    return root, project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'smoke', 'full', 'analyze'], default='smoke')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.stage in ('smoke', 'full') and (args.gpu is None or args.gpu < 0):
        parser.error('--gpu requires a nonnegative physical GPU index')
    root = BASE / 'evaluation_runs/self_vot22_ep0025'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, project = prepare(BASE)
        if args.stage == 'prepare':
            print('PREPARED; no tracking has been started', flush=True)
            return
        env = os.environ.copy()
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        env['PYTHONPATH'] = str(root) + os.pathsep + str(project) + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONUNBUFFERED'] = '1'
        if args.stage in ('smoke', 'full'):
            free = subprocess.check_output([
                'nvidia-smi', '-i', str(args.gpu), '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            if int(free) < 4096:
                raise RuntimeError('GPU {} has only {} MiB free; choose another GPU'.format(args.gpu, free))
            env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            workspace = root / args.stage
            cmd = [sys.executable, '-u', '-m', 'vot', 'evaluate', '--workspace', str(workspace), TRACKER]
            subprocess.run(cmd, cwd=workspace, env=env, check=True)
            print('VOT {} command exited 0; inspect VOT completion and tracker logs.'.format(args.stage), flush=True)
        else:
            workspace = root / 'full'
            for fmt in ('json', 'html'):
                cmd = [sys.executable, '-u', '-m', 'vot', 'analysis', '--workspace', str(workspace),
                       '--nocache', '--workers', '1', '--format', fmt, '--name', 'self_vot22_ep0025', TRACKER]
                subprocess.run(cmd, cwd=workspace, env=env, check=True)
            print('Reports:', workspace / 'analysis/self_vot22_ep0025', flush=True)


if __name__ == '__main__':
    main()
