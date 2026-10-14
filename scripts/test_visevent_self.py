"""Sequential VisEvent evaluation with an explicit checkpoint and isolated results."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    base = Path('/nas/liziyan103/seatrack')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    project = base / 'SEATrack'
    data = base / 'datasets/visevent/test_subset'
    checkpoint = base / 'training_runs/20261011_remaining/rgbe/models/checkpoints/rgbe/SEATrack_ep0045.pth.tar'
    variant = 'self_rgbe_ep0045'
    output = project / 'RGBE_workspace' / variant
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    manifest = dict(checkpoint=str(checkpoint), sha256=digest.hexdigest(),
                    data=str(data), project=str(project))
    manifest_path = output / 'checkpoint_used.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Existing results use different inputs'
    else:
        assert not list((output / 'VisEvent').glob('*.txt')), 'Unverified existing trajectories'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    os.chdir(project)
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location('visevent_runner', project / 'RGBE_workspace/test_rgbe_mgpus.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_parameters = runner.rgbe_prompt_params.parameters

    def self_parameters(*positional, **keywords):
        params = original_parameters(*positional, **keywords)
        params.checkpoint = str(checkpoint)
        print('Actual checkpoint:', params.checkpoint, flush=True)
        return params

    runner.rgbe_prompt_params.parameters = self_parameters
    names = sorted(s.strip() for s in (data / 'testlist.txt').read_text().splitlines() if s.strip())
    reference = base / 'tools/visevent_benchmark/VisEvent_testing_subset.txt'
    official_names = {s.strip() for s in reference.read_text().splitlines() if s.strip()}
    assert len(names) == len(set(names)) == 320 and set(names) == official_names
    selected = names[:1] if args.smoke else names
    import numpy as np
    started = time.time()
    for index, name in enumerate(selected, 1):
        runner.run_sequence(name, str(data), 'VisEvent', 'rgbe', num_gpu=1,
                            epoch=45, debug=0, script_name='seatrack', variants=variant,
                            hmoe_rank='4', amglora_rank='8')
        values = np.loadtxt(output / 'VisEvent' / (name + '.txt'), delimiter=',', ndmin=2)
        assert values.shape[1] == 4 and np.isfinite(values).all() and (values[:, 2:] > 0).all(), name
        print('Progress: {}/{} ({:.2f}%) | {} | elapsed {:.0f}s'.format(
            index, len(selected), index / len(selected) * 100, name, time.time() - started), flush=True)
    print('SMOKE PASSED' if args.smoke else 'ALL 320 SEQUENCES FINISHED; run metrics next', flush=True)


if __name__ == '__main__':
    main()
