import torch
from accumulation import Accumulation
from torch.cuda.amp import GradScaler

torch.set_num_threads(2)
torch.manual_seed(4)
a=torch.nn.Linear(3,2).double();b=torch.nn.Linear(3,2).double();b.load_state_dict(a.state_dict())
oa=torch.optim.AdamW(a.parameters(),lr=.0004);ob=torch.optim.AdamW(b.parameters(),lr=.0004)
acc=Accumulation(ob,GradScaler(enabled=False),b.parameters(),16,.1)
for iteration in range(2):
 x=torch.randn(64,3,dtype=torch.float64);y=torch.randn(64,2,dtype=torch.float64)
 oa.zero_grad(set_to_none=True);torch.nn.functional.mse_loss(a(x),y).backward()
 torch.nn.utils.clip_grad_norm_(a.parameters(),.1);oa.step()
 for i in range(16):
  stepped=acc.backward(torch.nn.functional.mse_loss(b(x[i*4:(i+1)*4]),y[i*4:(i+1)*4]))
  assert stepped == (i==15)
 acc.finish()
 for pa,pb in zip(a.parameters(),b.parameters()):assert torch.allclose(pa,pb,atol=1e-12,rtol=1e-10)
assert acc.updates==2
assert max(float(s['step']) for s in ob.state.values())==2
try:
 acc.backward(b(torch.ones(4,3,dtype=torch.float64)).square().mean());acc.finish()
except AssertionError:pass
else:raise AssertionError('Incomplete group not detected')
print('PASS: accumulation equals full batch for deterministic mean loss; clipping/AdamW/step counts/partial group checks')
