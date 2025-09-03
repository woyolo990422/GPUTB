import torch

class SKint(object):
    def __init__(self, dtype, device):
        self.rot_type = dtype
        self.device = device
        self.ss = ss 
        self.sp = sp 
        self.sd = sd 
        self.pp = pp 
        self.pd = pd 
        self.dd = dd 
        
        self.ss = torch.jit.script(self.ss)
        self.sp = torch.jit.script(self.sp)
        self.sd = torch.jit.script(self.sd)
        self.pp = torch.jit.script(self.pp)
        self.pd = torch.jit.script(self.pd)
        self.dd = torch.jit.script(self.dd)

        self.table = [[self.ss, self.sp, self.sd],
                      [self.sp, self.pp, self.pd],
                      [self.sd, self.pd, self.dd]]


    def rot_HS(self, angular0, angular1, Hvalue, Angvec):

        hs = self.table[angular0][angular1](Angvec, Hvalue).type(self.rot_type)
        hs.to(self.device)

        return hs


def ss(Angvec, SKss):
    # ss orbital no angular dependent.
    return SKss.unsqueeze(-1)


def sp(Angvec, SKsp):
    # rot_mat = Angvec[[1,2,0]].reshape([3 ,1])
    Angvec_=Angvec[:,[1, 2, 0]]
    hs = torch.einsum("ij,ik->ijk", Angvec_, SKsp)
    #hs = Angvec.view(-1) * SKsp.view(-1)
    return hs


def sd(Angvec, SKsd):
    x = Angvec[:,0]
    y = Angvec[:,1]
    z = Angvec[:,2]

    s3 = 3**0.5

    rot_mat = torch.stack([s3 * x * y, s3 * y * z, 1.5 * z**2 - 0.5,
                           s3 * x * z, s3 * (2.0 * x ** 2 - 1.0 + z ** 2) / 2.0],dim=1)
    
    hs = torch.einsum("ij,ik->ijk", rot_mat, SKsd)
    return hs


def pp(Angvec, SKpp):

    Angvec = Angvec[:,[1, 2, 0]].unsqueeze(-1)
    Angvec_T = Angvec.permute(0,2,1)
    mat = torch.bmm(Angvec,Angvec_T)
    rot_mat = torch.stack([mat, torch.eye(3, device=mat.device, dtype=mat.dtype)-mat], dim=-1)

    hs = torch.einsum("ijkl,il->ijk", rot_mat, SKpp)

    return hs


def pd(Angvec, SKpd):
    p = Angvec[:,[1, 2, 0]]
    x, y, z = Angvec[:,0], Angvec[:,1], Angvec[:,2]
    s3 = 3**0.5
    d = torch.stack([s3*x*y, s3*y*z, 0.5*(3*z*z-1), s3 *x*z, 0.5*s3*(x*x-y*y)],dim=-1)
    
    pd = torch.einsum("ij,ik->ijk", d, p)
    fm = torch.stack([x, 0*x, y, z, y, 0*x, -s3/3*y, 2*s3/3*z, -s3/3*x, 0*x, x, z, -y, 0*x, x],dim=-1).reshape(-1,5,3)
    rot_mat = torch.stack([pd, fm-2*s3/3*pd], dim=-1)

    hs = torch.einsum("ijkl,il->ijk", rot_mat, SKpd)

    return hs


def dd(Angvec, SKdd):

    x, y, z = Angvec[:,0], Angvec[:,1], Angvec[:,2]
    x2, y2, z2 = x**2, y**2, z**2
    xy, yz, zx = x*y, y*z, z*x
    s3 = 3**0.5
    d = torch.stack([s3*xy, s3*yz, 0.5*(3*z2-1), s3 * zx, 0.5*s3*(x2-y2)],dim=-1).reshape(-1,5,1)
    d_T=d.permute(0,2,1)

    dd0 = torch.bmm(d,d_T)
    dd2 = 1/3 * dd0
    dd2 = dd2 + torch.stack([
        z2, -zx, 2/s3*xy, -yz, 0*x,
        -zx, x2, -s3/3*yz, -xy, yz,
        2/s3*xy, -s3/3*yz, 2/3-z2, -s3/3*zx, s3/3*(x2-y2),
        -yz, -xy, -s3/3*zx, y2, -zx,
        0*x, yz, s3/3*(x2-y2), -zx, z2
    ],dim=-1).reshape(-1,5, 5)
    rot_mat = torch.stack(
        [dd0, torch.eye(5, device=dd0.device, dtype=dd0.dtype)-dd0-dd2, dd2], dim=-1)

    hs = torch.einsum("ijkl,il->ijk", rot_mat, SKdd)

    return hs
