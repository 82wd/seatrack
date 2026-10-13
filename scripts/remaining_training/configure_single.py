from pathlib import Path
import shutil, hashlib, json, ast
RUN=Path('/nas/liziyan103/seatrack/training_runs/20261011_remaining')
SRC=RUN/'code'; CODE=RUN/'code_single'
assert not CODE.exists(), 'Refusing to overwrite an existing single-GPU snapshot'
shutil.copytree(SRC,CODE,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(RUN/'scripts/accumulation.py',CODE/'lib/train/trainers/accumulation.py')
p=CODE/'experiments/seatrack/rgbt.yaml';s=p.read_text()
s=s.replace('BATCH_SIZE: 32','BATCH_SIZE: 4').replace('PRINT_INTERVAL: 50','PRINT_INTERVAL: 160')
# Match the 59968 examples actually consumed by the original two-rank drop_last loader.
s=s.replace('    SAMPLE_PER_EPOCH: 60000','    SAMPLE_PER_EPOCH: 59968',1)
p.write_text(s)
p=CODE/'lib/train/train_script.py';s=p.read_text().replace('    update_settings(settings, cfg)','    update_settings(settings, cfg)\n    settings.accumulation_steps = 16')
p.write_text(s)
p=CODE/'lib/train/trainers/ltr_trainer.py';s=p.read_text()
s=s.replace('from lib.utils.misc import get_world_size','from lib.utils.misc import get_world_size\nfrom .accumulation import Accumulation')
s=s.replace('        self._init_timing()','''        self._init_timing()
        accumulation_steps = getattr(self.settings, 'accumulation_steps', 1)
        if loader.training:
            assert len(loader) % accumulation_steps == 0
            accumulator = Accumulation(self.optimizer, self.scaler,
                                       self.actor.net.parameters(), accumulation_steps,
                                       self.settings.grad_clip_norm)
            print('Effective batch: 64; microbatch: 4; accumulation: 16; updates/epoch:', len(loader)//accumulation_steps, flush=True)''',1)
a=s.index('            if loader.training:\n                self.optimizer.zero_grad()');b=s.index('            # update statistics',a)
s=s[:a]+'''            if loader.training:
                accumulator.backward(loss)

'''+s[b:]
s=s.replace('        # calculate ETA after every epoch','''        if loader.training:
            accumulator.finish()
            print('Optimizer updates this epoch:', accumulator.updates, flush=True)
            print('CUDA peak allocated MiB:', torch.cuda.max_memory_allocated()/1024**2,
                  'reserved MiB:', torch.cuda.max_memory_reserved()/1024**2, flush=True)

        # calculate ETA after every epoch''')
p.write_text(s)
p=CODE/'lib/train/run_training.py';s=p.read_text();s=s.replace('    args = parser.parse_args()','''    args = parser.parse_args()
    if os.environ.get('SEATRACK_MEMORY_LIMIT_MB'):
        device_total = torch.cuda.get_device_properties(0).total_memory
        fraction = int(os.environ['SEATRACK_MEMORY_LIMIT_MB']) * 1024**2 / device_total
        torch.cuda.set_per_process_memory_fraction(min(fraction, 0.95), 0)''')
p.write_text(s)
for p in CODE.rglob('*.py'):ast.parse(p.read_text())
(RUN/'code_single_manifest.json').write_text(json.dumps({str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.rglob('*') if p.is_file()},indent=2))
(RUN/'rgbt/single_gpu_config.json').write_text(json.dumps(dict(microbatch=4,accumulation=16,effective_batch=64,epochs=60,updates_per_epoch=937,consumed_samples_per_epoch=59968,amp=False,lr=0.0004,code=str(CODE)),indent=2))
print('SINGLE GPU SNAPSHOT READY',CODE)
