import torch
import math
from torch_geometric.utils import scatter
from utilities.nep_utilities import find_fc, find_fn
ACT=torch.nn.functional.silu

class NEPNN(torch.nn.Module):
    def __init__(self,para):
        super().__init__()
        num_type=para.num_types
        self.n_max=para.n_max
        self.basis_size=para.basis_size
        embedding_init_val_0=para.embedding_init_val_0
        embedding_init_val_1=para.embedding_init_val_1

        des_hidden=para.des_hidden
        
        self.c_radial_ons_1D_0=torch.nn.Parameter(torch.empty(num_type*num_type, des_hidden ,   self.basis_size,dtype=torch.float32), requires_grad=True)
        self.c_radial_ons_1D_1=torch.nn.Parameter(torch.empty(num_type*num_type,  self.n_max ,   des_hidden,dtype=torch.float32), requires_grad=True)

        self.c_radial_hop_0=torch.nn.Parameter(torch.empty(num_type*num_type, des_hidden , self.basis_size, dtype=torch.float32), requires_grad=True)
        self.c_radial_hop_1=torch.nn.Parameter(torch.empty(num_type*num_type, self.n_max , des_hidden, dtype=torch.float32), requires_grad=True)

        torch.nn.init.normal_(self.c_radial_hop_0, mean=0, std=embedding_init_val_0)
        torch.nn.init.normal_(self.c_radial_hop_1, mean=0, std=embedding_init_val_0)
        torch.nn.init.normal_(self.c_radial_ons_1D_0, mean=0, std=embedding_init_val_1)
        torch.nn.init.normal_(self.c_radial_ons_1D_1, mean=0, std=embedding_init_val_1)

        self.bn1=torch.nn.BatchNorm1d(des_hidden,track_running_stats=False)
        self.bn2=torch.nn.BatchNorm1d(self.n_max ,track_running_stats=False)

    def forward(self, net_type: str, batch_ij, batch_fn: torch.Tensor,norm):
        if net_type == "radial_hop":

            output = torch.bmm(self.c_radial_hop_0[batch_ij],batch_fn)
            if norm:
                output = self.bn1(output)
            output = ACT(output)
            output = torch.bmm(self.c_radial_hop_1[batch_ij],output)
            if norm:
                output = self.bn2(output)
            output = ACT(output)
            
        elif net_type == "radial_ons_1D":

            output = torch.bmm(self.c_radial_ons_1D_0[batch_ij],batch_fn)
            if norm:
                output = self.bn1(output)
            output = ACT(output)
            output = torch.bmm(self.c_radial_ons_1D_1[batch_ij],output)
            if norm:
                output = self.bn2(output)
            output = ACT(output)

        return output.reshape([-1, self.n_max])


class NEP_Simple(torch.nn.Module):
    def __init__(self,para,device="cuda"):
        super().__init__()
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            if device == "cuda":
                raise ValueError("No CUDA devices found.")
            
        self.nn_split_scale=para.nn_split_scale

        self.para=para
        self.annmb=NEPNN(para)
        self.annmb.to(self.device)
    

    def find_descriptors_radial_per_ij_hop(self, dataset):
        d12 = dataset.d_hop.chunk(self.nn_split_scale,dim=0)
        bond_type = (dataset.Atomtype[dataset.i_hop]*self.para.num_types+dataset.Atomtype[dataset.j_hop]).chunk(self.nn_split_scale,dim=0)
        q_per_ij_total=torch.empty(0,device=self.device)

        for i in range(len(d12)):
            fc12 = find_fc(torch.as_tensor(1/self.para.cutoff), d12[i])
            fn = find_fn(torch.as_tensor(self.para.basis_size),torch.as_tensor(1/self.para.cutoff), d12[i], fc12)
            bond_type_tmp=torch.as_tensor(bond_type[i],dtype=torch.int32)
            q_per_ij = self.annmb("radial_hop",bond_type_tmp,fn.reshape([-1,self.para.basis_size,1]),self.para.des_norm)
            q_per_ij_total=torch.cat((q_per_ij_total,q_per_ij),dim=0)
        return q_per_ij_total

    def find_descriptors_radial_per_ij_ons_1D(self, dataset):
        d12 = dataset.d_ons.chunk(self.nn_split_scale,dim=0)
        bond_type = (dataset.Atomtype[dataset.i_ons]*self.para.num_types+dataset.Atomtype[dataset.j_ons]).chunk(self.nn_split_scale,dim=0)
        q_per_ij_total=torch.empty(0,device=self.device)
        for i in range(self.nn_split_scale):
            fc12 = find_fc(torch.as_tensor(1/self.para.cutoff), d12[i])
            fn = find_fn(torch.as_tensor(self.para.basis_size),torch.as_tensor(1/self.para.cutoff), d12[i], fc12)
            bond_type_tmp=torch.as_tensor(bond_type[i],dtype=torch.int32)
            q_per_ij=self.annmb("radial_ons_1D",bond_type_tmp,fn.reshape([-1,self.para.basis_size,1]),self.para.des_norm)
            q_per_ij_total=torch.cat((q_per_ij_total,q_per_ij),dim=0)
        return q_per_ij_total
    
    def find_descriptors_radial_poly(self, dataset):
        n1=dataset.i_ons.to(torch.int64)
        Q_radial_per_ij=self.find_descriptors_radial_per_ij_ons_1D(dataset)
        q_radial=scatter(Q_radial_per_ij, n1, dim=0, reduce='sum')
        return q_radial