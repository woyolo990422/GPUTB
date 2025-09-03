import torch

@torch.jit.script
def find_fc(rcinv, d12):
    fc = 0.5*torch.cos((3.1415926535*rcinv)*d12)+0.5
    return fc

@torch.jit.script
def find_fn_loop(x,a,b):
   return 2.0*x*a-b

def find_fn(n, rcinv, d12, fc12):
    fn = torch.zeros(fc12.shape[0], n,device=fc12.device)
    x = 2.0*(d12*rcinv-1)**2-1.0
    fn[:, 0] = 1.0
    fn[:, 1] = x
    for m in range(2, n.item()):
        fn[:, m] = find_fn_loop(x,fn[:, m-1],fn[:, m-2]) #2.0*x*fn[:, m-1]-fn[:, m-2]

    return (fn+1)*0.5*fc12.reshape([-1, 1])
