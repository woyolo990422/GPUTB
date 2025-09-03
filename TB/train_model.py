import torch
from TB.NEP import NEP_Simple ,ACT
from TB.Parameters import Parameters
from utilities.Index import Index
from utilities.SK_utilities import SKint
from utilities.onsiteDB import onsite_energy_database
import numpy as np
torch.set_printoptions(precision=8,linewidth=2000)
import h5py
import os
import sys

class Edge_net(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, num_type=1,init_val=0.01):
        super().__init__()
        bond_type=int(num_type*(num_type+1)/2)
        self.layer1 = torch.nn.Parameter(torch.empty(bond_type,out_features,in_features,dtype=torch.float32),requires_grad=True)
        self.linear_adjust = torch.nn.Linear(in_features, out_features)
        torch.nn.init.constant_(self.linear_adjust.bias,0)
        torch.nn.init.normal_(self.layer1, mean=0, std=init_val)
        torch.nn.init.normal_(self.linear_adjust.weight, mean=0, std=init_val)

    def forward(self, batch_ij, batch_descriptor):
        output = torch.bmm(self.layer1[batch_ij], batch_descriptor)
        output = ACT(output)

        return output+self.linear_adjust(batch_descriptor.squeeze(-1)).unsqueeze(-1)

class Env_net(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, num_type=1,init_val=0.01):
        super().__init__()
        bond_type=int(num_type*(num_type+1)/2)
        self.layer1 = torch.nn.Parameter(torch.empty(bond_type,out_features,in_features,dtype=torch.float32),requires_grad=True)
        self.linear_adjust = torch.nn.Linear(in_features, out_features)
        torch.nn.init.constant_(self.linear_adjust.bias,0)
        torch.nn.init.normal_(self.layer1, mean=0, std=init_val)
        torch.nn.init.normal_(self.linear_adjust.weight, mean=0, std=init_val)


    def forward(self, batch_ij, batch_descriptor):
        output = torch.bmm(self.layer1[batch_ij], batch_descriptor)
        output = ACT(output)
        return output+self.linear_adjust(batch_descriptor.squeeze(-1)).unsqueeze(-1)

class Site_net(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, num_type=1,init_val=0.01):
        super().__init__()
        self.layer1 = torch.nn.Parameter(torch.empty(num_type,out_features,in_features,dtype=torch.float32),requires_grad=True)
        self.linear_adjust = torch.nn.Linear(in_features, out_features)
        torch.nn.init.constant_(self.linear_adjust.bias,0)
        torch.nn.init.normal_(self.layer1, mean=0, std=init_val)
        torch.nn.init.normal_(self.linear_adjust.weight, mean=0, std=init_val)


    def forward(self, batch_i, batch_descriptor):
        output = torch.bmm(self.layer1[batch_i], batch_descriptor.unsqueeze(-1))
        output = ACT(output).squeeze(-1)
        return output +self.linear_adjust(batch_descriptor)


class Train_Model(torch.nn.Module):
    def __init__(self, para: Parameters, index: Index,dataloader):
        super().__init__()
        self.para = para
        self.Index = index
        self.dtype = torch.float32
        self.complexdtype = torch.complex64
        self.device = self.para.device
        self.energy_level = self.para.energy_level
        self.dataloader=dataloader
        self.frame_offset,self.resort_hop_index,self.resort_hop=[],[],[]
        self.orbit_length=[]
        for i in range(len(self.Index.orbit)):
            self.orbit_length.append(torch.sum(torch.as_tensor(self.Index.orbit[i])*2+1).item())
        self.orbit_length=torch.as_tensor(self.orbit_length,device=self.device)


        for data_index,data  in enumerate (self.dataloader):
            resort_hop_index,resort_hop=self.sort_type_hop(data)
            self.resort_hop_index.append(resort_hop_index)
            self.resort_hop.append(resort_hop)

        # self.resort_hop = [self.resort_hop[i] for i in indices]

        # self.resort_hop_index = [self.resort_hop_index[i] for i in indices]


        self.SKint = SKint(torch.float32, self.device)
        self.HoppingParamNum = self.Index.HoppingParamNum
        self.HoppingIndex_ptr = self.Index.HoppingIndex_ptr

        self.OnsiteEnergy, self.OnsiteEnergy_offset = self.find_OnsitesEnergy()

        self.nep = NEP_Simple(self.para,device=self.device)
        # self.nep_env = NEP_Simple(self.para,device=self.device)

                        
        list_length = self.para.num_types

        out_feature_2DList_hop = [[0 for j in range(list_length)]
                              for i in range(list_length)]
        for i in range(list_length):
            for j in range(list_length):
                index_start, index_end = self.HoppingIndex_ptr[i][j]
                out_feature_2DList_hop[i][j] = index_end-index_start
        
        num_outfeture = np.max(out_feature_2DList_hop).item()


        out_feature_1DList = [0 for i in range(list_length)]

        for atomtype in self.Index.orbit:
            num_orb=len(self.Index.orbit[atomtype])
            out_feature_1DList[atomtype] = num_orb

        max_feature=max(out_feature_1DList)

        self.Des_to_Edge_net = Edge_net(self.para.n_max,num_outfeture,list_length,self.para.param_init_val_0).to(self.device)
        self.Des_to_Site_net = Site_net(self.para.n_max,max_feature,len(self.Index.orbit),self.para.param_init_val_1).to(self.device)
        self.Des_to_Env_net = Env_net(self.para.n_max,self.para.n_max,list_length,self.para.param_init_val_0).to(self.device)
        
        
        self.Des_to_Env_net = torch.jit.script(self.Des_to_Env_net)
        self.Des_to_Site_net=torch.jit.script(self.Des_to_Site_net)
        self.Des_to_Edge_net=torch.jit.script(self.Des_to_Edge_net)

        self.mapping = torch.tensor([0, 1, 2,-1,3,4,-1,-1,5,6,7,-1,-1,8,-1,-1-1,-1,9],device=self.device)

    def find_OnsitesEnergy(self):
        Atom_to_Symbol = dict(
            zip(self.Index.Symbol_to_AtomType.values(), self.Index.Symbol_to_AtomType.keys()))
        onsite, offset = [], [0]
        for atomtype in self.Index.orbit:
            symbol = Atom_to_Symbol[atomtype]
            orbit_num = 0
            for nl_index, nl in enumerate(self.Index.orbit_input[symbol]):
                oe = onsite_energy_database[symbol][nl]+self.energy_level[symbol][nl]/27.211324570274
                onsite.append(oe)
                orbit_num += 1
            offset.append(orbit_num+offset[-1])
        return torch.tensor(onsite, requires_grad=False), offset


    def find_hoppings_and_onsites(self, data,batch_index):

        resort_hop=self.resort_hop[batch_index]
        
        Des_radial_hop_ = self.nep.find_descriptors_radial_per_ij_hop(data)[resort_hop]

        self.Des_radial_ons_1D = self.nep.find_descriptors_radial_poly(data)

        i_hop = data.edge_index_hop[0][resort_hop]
        j_hop = data.edge_index_hop[1][resort_hop]
        atomtype_i_hop = data.Atomtype[i_hop]
        atomtype_j_hop = data.Atomtype[j_hop]

        bond_type_1=self.mapping[atomtype_i_hop*atomtype_i_hop+atomtype_j_hop*atomtype_j_hop]

        Des=self.Des_radial_ons_1D[i_hop]+self.Des_radial_ons_1D[j_hop]         #torch.cat((self.Des_radial_ons_1D[i_hop],self.Des_radial_ons_1D[j_hop]),dim=1)
        
        Des_e=self.Des_to_Env_net(bond_type_1,Des[...,None]).squeeze(-1)

        self.hoppings = self.Des_to_Edge_net(bond_type_1,(Des_e*Des_radial_hop_)[...,None]).squeeze(-1)

    def sort_type_hop(self,data):
        Atomtype=data.Atomtype
        i_hop=data.edge_index_hop[0]
        j_hop=data.edge_index_hop[1]
        resort_hop_=[]
        resort_hop_index_=[0]
        length_offset_hop=0
        for type_i in range(self.para.num_types):
            for type_j in range(type_i,self.para.num_types):
                hop_index_seleced = torch.arange(len(i_hop),dtype=torch.long, device=self.device)\
                    [(Atomtype[i_hop] == type_i)*(Atomtype[j_hop] == type_j)]
                resort_hop_.append(hop_index_seleced)
                length_offset_hop+=len(hop_index_seleced)
                resort_hop_index_.append(length_offset_hop)

        resort_hop = torch.as_tensor([item for sublist in resort_hop_ for item in sublist],device=self.device)
        return resort_hop_index_,resort_hop
    

    def find_realspace_hamiltonian_hoppingblock(self, data,batch_index):
        
        resort_hop_index=self.resort_hop_index[batch_index]

        resort_hop=self.resort_hop[batch_index]

        directional_cosine=(data.D_hop/data.d_hop.unsqueeze(-1)).to(device=self.device)[resort_hop]

        type_hop_offset=0

        hopping_matrixes = []

        for type_i in range(self.para.num_types):
            
            for type_j in range(type_i,self.para.num_types):
                
                amnum_i = len(self.Index.orbit[type_i])
                amnum_j = len(self.Index.orbit[type_j])
                
                resort_hop_index_s=resort_hop_index[type_hop_offset]
                resort_hop_index_e=resort_hop_index[type_hop_offset+1]

                hop_length=resort_hop_index_e-resort_hop_index_s

                tmp_block=torch.zeros(hop_length,self.orbit_length[type_i],self.orbit_length[type_j], dtype=torch.float32, device=self.device)

                tmp_hoppings=self.hoppings[resort_hop_index_s:resort_hop_index_e]

                directional_cosine_=directional_cosine[resort_hop_index_s:resort_hop_index_e]

                hopping_offset=0

                if type_i==type_j:

                    offset_i = 0

                    for am_i_index in range(amnum_i):
                    
                        am_i = self.Index.orbit[type_i][am_i_index]

                        orbitnum_i = 2*am_i+1

                        offset_j = offset_i

                        for am_j_index in range(am_i_index,amnum_j):

                            am_j = self.Index.orbit[type_j][am_j_index]

                            orbitnum_j = 2*am_j+1

                            hoppingparamnum = min(am_i, am_j)+1

                            hopping_param = tmp_hoppings[:,hopping_offset:hopping_offset+hoppingparamnum]

                            sk_block = self.SKint.rot_HS(am_i, am_j, hopping_param, directional_cosine_)

                            if (am_i_index == am_j_index):

                                tmp_block[:,offset_i:offset_i+orbitnum_i,offset_j:offset_j+orbitnum_j] = sk_block

                            elif (am_i <= am_j):

                                tmp_block[:,offset_i:offset_i+orbitnum_i,
                                                offset_j:offset_j+orbitnum_j] = (-1)**(am_i+am_j)*torch.transpose(sk_block, dim0=1, dim1=2)
                                tmp_block[:,offset_j:offset_j+orbitnum_j,
                                               offset_i:offset_i+orbitnum_i] = sk_block
                            elif (am_i > am_j):

                                tmp_block[:,offset_i:offset_i+orbitnum_i,
                                                offset_j:offset_j+orbitnum_j] = sk_block
                                tmp_block[:,offset_j:offset_j+orbitnum_j,
                                               offset_i:offset_i+orbitnum_i] = (-1)**(am_i+am_j)*torch.transpose(sk_block, dim0=1, dim1=2)

                            hopping_offset+=hoppingparamnum


                            offset_j += orbitnum_j
                        offset_i += orbitnum_i
                    type_hop_offset+=1

                elif type_i<type_j:
                    offset_i = 0

                    for am_i_index in range(amnum_i):
                    
                        am_i = self.Index.orbit[type_i][am_i_index]

                        orbitnum_i = 2*am_i+1

                        offset_j = 0
                        for am_j_index in range(amnum_j):

                            am_j = self.Index.orbit[type_j][am_j_index]

                            orbitnum_j = 2*am_j+1

                            hoppingparamnum = min(am_i, am_j)+1

                            hopping_param = tmp_hoppings[:,hopping_offset:hopping_offset+hoppingparamnum]

                            sk_block = self.SKint.rot_HS(am_i, am_j, hopping_param, directional_cosine_)

                            if (am_i <= am_j):
                                tmp_block[:,offset_i:offset_i+orbitnum_i,
                                                offset_j:offset_j+orbitnum_j] = (-1)**(am_i+am_j)*torch.transpose(sk_block, dim0=1, dim1=2)
                            else:
                                tmp_block[:,offset_i:offset_i+orbitnum_i,
                                                offset_j:offset_j+orbitnum_j] = sk_block

                            hopping_offset+=hoppingparamnum

                            offset_j += orbitnum_j
                        offset_i += orbitnum_i
                    type_hop_offset+=1

                hopping_matrixes.append(tmp_block)

        self.hopping_matrixes=[tensor[i] for tensor in hopping_matrixes for i in range(tensor.size(0))]
        
    def find_realspace_hamiltonian_onsiteblock(self, data):

        Des_radial_ons_1D_val = self.Des_to_Site_net(data.Atomtype, self.Des_radial_ons_1D).unsqueeze(-1)
        
        frame_num_atoms=data.num_nodes if data.num_nodes<100 else data[0].num_nodes##TODO

        self.onsite_matrixes=[]

        for atom_index in range(frame_num_atoms):
            
            if self.para.batch_size==1:
                atomtype=data.Atomtype[atom_index].item()
                ons_1D=Des_radial_ons_1D_val[atom_index+frame_num_atoms*torch.arange(1)]
                onsite_energy_list = torch.zeros(1,sum(torch.as_tensor(self.Index.orbit[atomtype])*2+1).item(),device=self.device)
            else:
                atomtype=data[0].Atomtype[atom_index].item()
                ons_1D=Des_radial_ons_1D_val[atom_index+frame_num_atoms*torch.arange(data.num_graphs)]
                onsite_energy_list = torch.zeros(data.num_graphs,sum(torch.as_tensor(self.Index.orbit[atomtype])*2+1).item(),device=self.device)

        
            offset = self.OnsiteEnergy_offset[atomtype]

            orbitnum_am=0

            for am_index, am in enumerate(self.Index.orbit[atomtype]):

                oe = self.OnsiteEnergy[offset]

                oe2=ons_1D[:,am_index]

                orbitnum_am_plus = 2*am+1

                onsite_energy_list[:,orbitnum_am:orbitnum_am+orbitnum_am_plus]=(oe+oe2)

                orbitnum_am += orbitnum_am_plus

                offset+=1

            onsite_matrix=torch.einsum('ij,jk->ijk', onsite_energy_list, torch.eye(onsite_energy_list.shape[1], dtype=torch.float32, device=self.device))

            self.onsite_matrixes.append(onsite_matrix)

    def find_kspace_hamiltonian(self, data,batch_index):

        resort_hop=self.resort_hop[batch_index]

        if self.para.batch_size==1:

            Hks = torch.zeros((1,data.kpoints.shape[0],data.total_orbit_num, data.total_orbit_num),dtype=torch.complex64, device=self.device)
            ft_factor = torch.einsum('ij,kj->ik', data.kpoints, data.S_hop[resort_hop].float())
            ft_factor_edge=torch.exp(-2.j*torch.pi*ft_factor).unsqueeze(-1).unsqueeze(-1)

            i_total = data.edge_index_hop[0][resort_hop]
            j_total = data.edge_index_hop[1][resort_hop]

            orbit_offset=torch.as_tensor(data.orbit_offset,device=self.device)
            orbit_num_per_atom=torch.as_tensor(data.orbit_num_per_atom,device=self.device)

            offset_i = orbit_offset[i_total]
            offset_j = orbit_offset[j_total]

            orbit_num_i = orbit_num_per_atom[i_total]
            orbit_num_j = orbit_num_per_atom[j_total]

            for AtomPair_index in range(data.num_edges):
                Hks[0,:,offset_i[AtomPair_index]:offset_i[AtomPair_index]+orbit_num_i[AtomPair_index], offset_j[AtomPair_index]:offset_j[AtomPair_index]+orbit_num_j[AtomPair_index]] +=  self.hopping_matrixes[AtomPair_index] * ft_factor_edge[:, AtomPair_index]
            
            num_atoms=data.num_nodes

            onsite_offset = 0

            for atom_index in range(num_atoms):

                onsite_matrixe=self.onsite_matrixes[atom_index].unsqueeze(1)
                
                orbit_num = data.orbit_num_per_atom[atom_index]
    
                Hks[:,:,onsite_offset:onsite_offset+orbit_num,onsite_offset:onsite_offset+orbit_num] += 0.5*onsite_matrixe

                onsite_offset += orbit_num

            self.kspace_hamiltonians = Hks+Hks.permute(0,1,3,2).conj()

        else:
            Hks = torch.zeros((data.num_graphs,data[0].kpoints.shape[0],data[0].total_orbit_num, data[0].total_orbit_num),dtype=torch.complex64, device=self.device)
            ft_factor = torch.einsum('ij,kj->ik', data[0].kpoints, data.S_hop[resort_hop].float())
            ft_factor_edge=torch.exp(-2.j*torch.pi*ft_factor).unsqueeze(-1).unsqueeze(-1)

            i_total = data.edge_index_hop[0][resort_hop]
            j_total = data.edge_index_hop[1][resort_hop]

            frame_total=data.frame_offset[resort_hop]-batch_index*self.para.nepin["batch_size"]

            orbit_offset=torch.as_tensor(data.orbit_offset,device=self.device)
            orbit_num_per_atom=torch.as_tensor(data.orbit_num_per_atom,device=self.device)

            offset_i = orbit_offset[i_total]
            offset_j = orbit_offset[j_total]

            orbit_num_i = orbit_num_per_atom[i_total]
            orbit_num_j = orbit_num_per_atom[j_total]


            for AtomPair_index in range(data.num_edges):
                Hks[frame_total[AtomPair_index],:,offset_i[AtomPair_index]:offset_i[AtomPair_index]+orbit_num_i[AtomPair_index], offset_j[AtomPair_index]:offset_j[AtomPair_index]+orbit_num_j[AtomPair_index]] +=  self.hopping_matrixes[AtomPair_index] * ft_factor_edge[:, AtomPair_index]

            num_atoms=data[0].num_nodes

            onsite_offset = 0

            for atom_index in range(num_atoms):

                onsite_matrixe=self.onsite_matrixes[atom_index].unsqueeze(1)
                
                orbit_num = data[0].orbit_num_per_atom[atom_index]
    
                Hks[:,:,onsite_offset:onsite_offset+orbit_num,onsite_offset:onsite_offset+orbit_num] += 0.5*onsite_matrixe

                onsite_offset += orbit_num

            self.kspace_hamiltonians = Hks+Hks.permute(0,1,3,2).conj()

    def find_kspace_hamiltonian_large(self, data,batch_index):

        resort_hop=self.resort_hop[batch_index].to("cpu")

        k_num=data.kpoints.shape[0]

        ft_factor = torch.einsum('ij,kj->ik', data.kpoints, data.S_hop[resort_hop].float()).to("cpu")
        ft_factor_edge=torch.exp(-2.j*torch.pi*ft_factor).unsqueeze(-1).unsqueeze(-1)

        i_total = data.edge_index_hop[0][resort_hop].to("cpu")
        j_total = data.edge_index_hop[1][resort_hop].to("cpu")

        orbit_offset=torch.as_tensor(data.orbit_offset,device="cpu")
        orbit_num_per_atom=torch.as_tensor(data.orbit_num_per_atom,device="cpu")

        offset_i = orbit_offset[i_total].to("cpu")
        offset_j = orbit_offset[j_total].to("cpu")

        orbit_num_i = orbit_num_per_atom[i_total].to("cpu")
        orbit_num_j = orbit_num_per_atom[j_total].to("cpu")
        
        block_indices = torch.arange(k_num)

        x_ranges = [torch.arange(offset_i[idx], offset_i[idx] + orbit_num_i[idx]) 
                for idx in range(data.num_edges)]
        
        y_ranges = [torch.arange(offset_j[idx], offset_j[idx] + orbit_num_j[idx]) 
                for idx in range(data.num_edges)]

        matrix_val = [self.hopping_matrixes[AtomPair_index].to("cpu") * ft_factor_edge[:, AtomPair_index] for  AtomPair_index in range(data.num_edges)]

        all_indices = []
        all_values = []

        for AtomPair_index in range(data.num_edges):#TODO 改为按照type放置 速度更快

            x_indicate=x_ranges[AtomPair_index]
            y_indicate=y_ranges[AtomPair_index]
            matrix_val_tmp=matrix_val[AtomPair_index].flatten()

            block_grid, row_grid, col_grid = torch.meshgrid(block_indices, x_indicate, y_indicate, indexing="ij")
            indices = torch.stack([block_grid.flatten(), row_grid.flatten(), col_grid.flatten()], dim=0)

            all_indices.append(indices)
            all_values.append(matrix_val_tmp)

        final_indices = torch.cat(all_indices, dim=1)
        final_values = torch.cat(all_values, dim=0)    
        HKs = torch.sparse_coo_tensor(final_indices, final_values, (k_num,data.total_orbit_num,data.total_orbit_num))

        num_atoms=data.num_nodes

        onsite_offset = 0

        all_indices = []
        all_values = []

        for atom_index in range(num_atoms):

            onsite_matrixe=self.onsite_matrixes[atom_index].unsqueeze(1).to("cpu").repeat(1, k_num, 1, 1)
            
            orbit_num = data.orbit_num_per_atom[atom_index]

            x_indicate=torch.arange(onsite_offset,onsite_offset+orbit_num)

            block_grid, row_grid, col_grid = torch.meshgrid(block_indices, x_indicate, x_indicate)

            indices = torch.stack([block_grid.flatten(), row_grid.flatten(), col_grid.flatten()], dim=0)

            all_indices.append(indices)
            all_values.append(onsite_matrixe.flatten())

            onsite_offset += orbit_num

        final_indices = torch.cat(all_indices, dim=1)
        final_values = torch.cat(all_values, dim=0)   

        HKs +=0.5* torch.sparse_coo_tensor(final_indices, final_values, (k_num,data.total_orbit_num,data.total_orbit_num))

        Hks_conjugated = HKs.permute(0, 2, 1).conj()

        self.kspace_hamiltonians = HKs+Hks_conjugated

        pass


    @torch.jit.script
    def find_eigs(kspace_hamiltonians:torch.Tensor):
        return torch.linalg.eigvalsh(kspace_hamiltonians)* 27.211324570274


    def find_Rspace_hamiltonian(self, data,batch_index):
        self.find_hoppings_and_onsites(data,batch_index)
        self.find_realspace_hamiltonian_hoppingblock(data,batch_index)
        self.find_realspace_hamiltonian_onsiteblock(data)

    def find_eigenvalues(self, data,batch_index):
        self.find_kspace_hamiltonian(data,batch_index)
        eigs_list=self.find_eigs(self.kspace_hamiltonians)
        return eigs_list
    
    def find_eigenvalues_large(self, data,batch_index):
        self.find_kspace_hamiltonian_large(data,batch_index)
        
        for i in range(self.kspace_hamiltonians.shape[0]):
            coo = self.kspace_hamiltonians[i].coalesce()
            row, col = coo.indices()[0].numpy(), coo.indices()[1].numpy()
            data = coo.values().numpy()

            file_path = os.path.join(self.para.error_figures_path, f"Ham_K{i}.h5")

            with h5py.File(file_path, 'w') as f:
                f.create_dataset('row', data=row)
                f.create_dataset('col', data=col)
                f.create_dataset('data', data=data)
                
            print(f"Sparse matrix {i} saved")
            sys.stdout.flush()
            
    def forward(self, data,batch_index):
        self.find_Rspace_hamiltonian(data,batch_index)
        if self.para.prediction==0 or self.para.prediction==1:
            eigs_list = self.find_eigenvalues(data,batch_index)
            return eigs_list
        elif self.para.prediction==2:
            self.find_eigenvalues_large(data,batch_index)
            #eigs_list=torch.linalg.eigvalsh(self.kspace_hamiltonians.to_dense())* 27.211324570274

            #return eigs_list.unsqueeze(0)
            sys.exit()
