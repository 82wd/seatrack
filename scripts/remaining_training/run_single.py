import os, sys, json, time, subprocess, fcntl, hashlib, traceback, re
from pathlib import Path
from datetime import datetime, timezone
RUN=Path('/nas/liziyan103/seatrack/training_runs/20261011_remaining')
CODE=RUN/'code_single'
TASK=sys.argv[1]
assert TASK in ('rgbt','rgbe')
OUT=RUN/TASK
sys.path.insert(0,str(CODE))
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'

def status(state, **details):
    record=dict(time=datetime.now(timezone.utc).isoformat(),state=state,task=TASK,**details)
    (OUT/'status.txt').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(record,flush=True)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def validate():
    import torch
    torch.set_num_threads(2)
    from lib.train.dataset.lasher import LasHeR
    from lib.train.dataset.visevent import VisEvent
    datasets=[LasHeR(split='train'),LasHeR(split='val')] if TASK=='rgbt' else [VisEvent()]
    from lib.config.seatrack.config import cfg,update_config_from_file
    update_config_from_file(str(CODE/f'experiments/seatrack/{TASK}.yaml'))
    minimum_visible = 2 * (getattr(cfg.DATA.SEARCH, 'NUMBER', 1) + getattr(cfg.DATA.TEMPLATE, 'NUMBER', 1))
    report=[]
    for dataset in datasets:
        eligible_count=0
        for i,name in enumerate(dataset.sequence_list):
            seq=Path(dataset._get_sequence_path(i));info=dataset.get_sequence_info(i)
            n=len(info['bbox'])
            assert n>0 and info['bbox'].shape==(n,4),str(seq)
            assert torch.isfinite(info['bbox']).all(),str(seq)+' nonfinite annotation'
            assert len(info['visible'])==n,str(seq)+' visibility length mismatch'
            eligible=int(info['visible'].sum())>minimum_visible and n>=20
            eligible_count+=int(eligible)
            folders=('visible','infrared') if TASK=='rgbt' else ('vis_imgs','event_imgs')
            ext='*.jpg' if TASK=='rgbt' else '*.bmp'
            lists=[list((seq/channel).glob(ext)) for channel in folders]
            assert all(len(x)==n for x in lists),(str(seq),n,list(map(len,lists)))
            if TASK=='rgbt':
                for files in lists:
                    ids=[int(re.search(r'(\d+)\.jpg$',x.name).group(1)) for x in files]
                    assert sorted(ids)==list(range(n)),str(seq)+' IDs must be 0..N-1'
            else:
                assert {x.name for x in lists[0]}=={x.name for x in lists[1]},str(seq)+' unpaired frames'
            for k in sorted({0,n//2,n-1}):
                img=dataset._get_frame(str(seq),k)
                assert img is not None and img.ndim==3 and img.shape[2]==6,(str(seq),k)
            report.append(dict(sequence=name,frames=n,visible=int(info['visible'].sum()),sampler_eligible=eligible))
            if i%50==0:print('validated',dataset.get_name(),i,name,flush=True)
        assert eligible_count>0,dataset.get_name()+' has no sampler-eligible sequences'
    (OUT/'data_validation.json').write_text(json.dumps(report,indent=2))
    from lib.config.seatrack.config import cfg,update_config_from_file
    update_config_from_file(str(CODE/f'experiments/seatrack/{TASK}.yaml'))
    weight=Path(cfg.MODEL.PRETRAIN_FILE)
    checkpoint=torch.load(weight,map_location='cpu')
    assert len(checkpoint['net'])==242
    assert not any('lora_' in k for k in checkpoint['net']), 'Expected OSTrack initialization, not trained SEATrack'
    from lib.models.seatrack.seatrack import build_seatrack
    model=build_seatrack(cfg,training=True)
    incompatible=model.load_state_dict(checkpoint['net'],strict=False)
    assert not incompatible.unexpected_keys,incompatible.unexpected_keys
    # All ordinary pretrained backbone/head weights must be present.
    allowed=('lora_', 'r2dte_', 'dte2r_', 'attn_moe', 'ffn_moe')
    bad=[k for k in incompatible.missing_keys if not any(a in k for a in allowed)]
    assert not bad,bad
    (OUT/'initialization.json').write_text(json.dumps(dict(path=str(weight),sha256=sha(weight),epoch=checkpoint.get('epoch'),missing=incompatible.missing_keys),indent=2))
    return int(cfg.TRAIN.EPOCH)

def idle():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
    available=[]
    for row in rows.splitlines():
        index,uuid,total,used=[x.strip() for x in row.split(',')]
        free=int(total)-int(used)
        if free>=6000:
            available.append((free,index,uuid))
    available.sort(reverse=True)
    return [(index,uuid) for free,index,uuid in available]

def acquire():
    previous=set();stable={};last_report=0
    while True:
        if TASK=='rgbt':
            try: predecessor=json.loads((RUN/'rgbe/status.txt').read_text()).get('state')
            except (ValueError,FileNotFoundError): predecessor=None
            if predecessor!='COMPLETED':
                time.sleep(30);continue
        if (RUN/'PAUSE').exists():
            time.sleep(30);continue
        try:
            candidates=idle();now=time.time()
            for gpu in candidates:
                stable[gpu]=stable.get(gpu,now) if gpu in previous else now
            previous=set(candidates)
            ready=[g for g in candidates if now-stable[g]>=60]
            if len(ready)>=1:
                guard=open(RUN/'gpu_selection.lock','a')
                with guard:
                    fcntl.flock(guard,fcntl.LOCK_EX)
                    locks=[];selected=[]
                    for g in ready:
                        lock=open(RUN/f'gpu_{g[0]}.lock','a')
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:lock.close();continue
                        locks.append(lock);selected.append(g)
                        if len(selected)==1:break
                    if len(selected)==1 and all(g in idle() for g in selected):return selected,locks
                    for lock in locks:lock.close()
            if now-last_report>1800:
                status('WAITING_GPU',available=[g[0] for g in candidates],required=1);last_report=now
        except Exception as e:
            if time.time()-last_report>1800:status('WAITING_GPU',query_error=str(e));last_report=time.time()
        time.sleep(30)

def main():
    if '--reuse-validation' in sys.argv:
        import yaml
        rows=json.loads((OUT/'data_validation.json').read_text())
        init=json.loads((OUT/'initialization.json').read_text())
        cfg=yaml.safe_load((CODE/f'experiments/seatrack/{TASK}.yaml').read_text())
        assert len(rows)==({'rgbt':979,'rgbe':500}[TASK])
        assert sha(cfg['MODEL']['PRETRAIN_FILE'])==init['sha256']
        epochs=int(cfg['TRAIN']['EPOCH'])
        status('VALIDATION_REUSED',sequences=len(rows),reason='Reuse unchanged dataset integrity and pretrained checksum; single-GPU accumulation config recorded separately')
    else:
        status('VALIDATING')
        epochs=validate()
    status('WAITING_GPU',required=1)
    selected,locks=acquire()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(g[1] for g in selected),PYTHONPATH=str(CODE),PYTHONUNBUFFERED='1',LOCAL_RANK='-1',SEATRACK_MEMORY_LIMIT_MB='5120')
    cmd=[sys.executable,'lib/train/run_training.py','--script','seatrack','--config',TASK,'--save_dir',str(OUT/'models'),'--seed','0']
    status('TRAINING',gpus=[g[0] for g in selected],command=cmd,epochs=epochs,batch_per_gpu=4,accumulation_steps=16,effective_batch=64,allocator_limit_mib=5120)
    with (OUT/'training.log').open('a',buffering=1) as log:
        result=subprocess.run(cmd,cwd=str(CODE),env=env,stdout=log,stderr=subprocess.STDOUT)
    for lock in locks:lock.close()
    assert result.returncode==0,'Training exit code '+str(result.returncode)
    import torch
    files=sorted((OUT/'models/checkpoints'/TASK).glob('SEATrack_ep*.pth.tar'))
    assert len(files)==epochs,(len(files),epochs)
    first=torch.load(files[0],map_location='cpu');last=torch.load(files[-1],map_location='cpu')
    assert first['epoch']==1 and last['epoch']==epochs
    changed=[k for k in first['net'] if not torch.equal(first['net'][k],last['net'][k])]
    assert changed,'No model tensors changed'
    def optimizer_step(checkpoint):
        values=[float(v['step']) for v in checkpoint['optimizer']['state'].values() if 'step' in v]
        assert values,'Missing optimizer step evidence'
        return max(values)
    first_step,last_step=optimizer_step(first),optimizer_step(last)
    assert first_step==937 and last_step==937*epochs,(first_step,last_step,epochs)
    evidence={'first_epoch':1,'last_epoch':epochs,'optimizer_steps':[first_step,last_step],'changed_tensors':len(changed),'changed_examples':changed[:10],'checkpoints':[dict(path=str(p),sha256=sha(p)) for p in files]}
    (OUT/'completion.json').write_text(json.dumps(evidence,indent=2))
    status('COMPLETED',epochs=epochs,checkpoints=len(files),changed_tensors=len(changed))

if __name__=='__main__':
    try:main()
    except Exception as e:
        status('FAILED',error=str(e));traceback.print_exc();sys.exit(1)
