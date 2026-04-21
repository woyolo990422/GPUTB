import torch
from LSQT.NEP_lsqt import NEP_Simple ,ACT
from TB.Parameters import Parameters
from utilities.Index import Index
from utilities.SK_utilities import SKint
from utilities.onsiteDB import onsite_energy_database
import numpy as np
import os
import h5py

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


        self.SKint = SKint(torch.float32, self.device)
        self.HoppingParamNum = self.Index.HoppingParamNum
        self.HoppingIndex_ptr = self.Index.HoppingIndex_ptr

        self.OnsiteEnergy, self.OnsiteEnergy_offset = self.find_OnsitesEnergy()

        self.nep = NEP_Simple(self.para,device=self.device)
        
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
        self.mapping = torch.tensor([0, 1, 2,-1,3,4,-1,-1,5,6,7,-1,-1,8,-1,-1-1,-1,9],dtype=torch.int32,device=self.device)


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

        Des_radial_hop_ = self.nep.find_descriptors_radial_per_ij_hop(data).chunk(self.para.nn_split_scale,dim=0)
      
        self.Des_radial_ons_1D = self.nep.find_descriptors_radial_poly(data)

        i_hop = data.i_hop.to(torch.int64)
        j_hop = data.j_hop.to(torch.int64)

        Des_i=self.Des_radial_ons_1D[i_hop].chunk(self.para.nn_split_scale,dim=0)
        Des_j=self.Des_radial_ons_1D[j_hop].chunk(self.para.nn_split_scale,dim=0)


        bond_type_1=(data.Atomtype[i_hop]*data.Atomtype[i_hop]+data.Atomtype[j_hop]*data.Atomtype[j_hop]).chunk(self.para.nn_split_scale)
        
        hoppings=torch.empty(0,device=self.device)

        for i in range(self.para.nn_split_scale):
            Des_radial_hop_tmp=Des_radial_hop_[i]

            bond_type_tmp=self.mapping[bond_type_1[i].to(torch.int32)]
            
            Des=Des_i[i]+Des_j[i]

            Des_e=self.Des_to_Env_net(bond_type_tmp,Des[...,None]).squeeze(-1)

            hoppings_tmp = self.Des_to_Edge_net(bond_type_tmp,(Des_e*Des_radial_hop_tmp)[...,None]).squeeze(-1)

            hoppings=torch.cat((hoppings,hoppings_tmp),dim=0)

        return hoppings

    def find_realspace_hamiltonian_hoppingblock(self, data,batch_index):
        
        hoppings=self.find_hoppings_and_onsites(data,batch_index)

        directional_cosine=(data.D_hop/data.d_hop.unsqueeze(-1))#.chunk(self.nn_split_scale,dim=0)
        
        resort_hop_index=data.resort_hop_index[0]

        type_hop_offset=0

        self.hopping_matrixes = []

        file_path = os.path.join(self.para.dump_sparseH, 'hoppings.h5')

        with h5py.File(file_path, 'w') as f:

            for type_i in range(self.para.num_types):
                
                for type_j in range(type_i,self.para.num_types):
                    
                    amnum_i = len(self.Index.orbit[type_i])
                    amnum_j = len(self.Index.orbit[type_j])
                    
                    resort_hop_index_s=resort_hop_index[type_hop_offset]
                    resort_hop_index_e=resort_hop_index[type_hop_offset+1]

                    hop_length=resort_hop_index_e-resort_hop_index_s
                    
                    tmp_block=torch.zeros(hop_length,data[0].orbit_num_per_atom[type_i],data[0].orbit_num_per_atom[type_j], dtype=torch.float32, device=self.device).chunk(self.para.nn_split_scale,dim=0)

                    tmp_hoppings=hoppings[resort_hop_index_s:resort_hop_index_e].chunk(self.para.nn_split_scale,dim=0)

                    directional_cosine_=directional_cosine[resort_hop_index_s:resort_hop_index_e].chunk(self.para.nn_split_scale,dim=0)

                    for split_index in range(self.para.nn_split_scale):
                        
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

                                    hopping_param = tmp_hoppings[split_index][:,hopping_offset:hopping_offset+hoppingparamnum]

                                    sk_block = self.SKint.rot_HS(am_i, am_j, hopping_param, directional_cosine_[split_index])


                                    if (am_i_index == am_j_index):

                                        tmp_block[split_index][:,offset_i:offset_i+orbitnum_i,offset_j:offset_j+orbitnum_j] = sk_block

                                    elif (am_i <= am_j):

                                        tmp_block[split_index][:,offset_i:offset_i+orbitnum_i,
                                                        offset_j:offset_j+orbitnum_j] = (-1)**(am_i+am_j)*torch.transpose(sk_block, dim0=1, dim1=2)
                                        tmp_block[split_index][:,offset_j:offset_j+orbitnum_j,
                                                    offset_i:offset_i+orbitnum_i] = sk_block
                                    elif (am_i > am_j):

                                        tmp_block[split_index][:,offset_i:offset_i+orbitnum_i,
                                                        offset_j:offset_j+orbitnum_j] = sk_block
                                        tmp_block[split_index][:,offset_j:offset_j+orbitnum_j,
                                                    offset_i:offset_i+orbitnum_i] = (-1)**(am_i+am_j)*torch.transpose(sk_block, dim0=1, dim1=2)

                                    hopping_offset+=hoppingparamnum

                                    offset_j += orbitnum_j
                                offset_i += orbitnum_i
                            #tmp_block[split_index]=torch.cat(tmp_block)
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

                                    hopping_param = tmp_hoppings[split_index][:,hopping_offset:hopping_offset+hoppingparamnum]

                                    sk_block = self.SKint.rot_HS(am_i, am_j, hopping_param, directional_cosine_[split_index])

                                    if (am_i <= am_j):
                                        tmp_block[split_index][:,offset_i:offset_i+orbitnum_i,
                                                        offset_j:offset_j+orbitnum_j] = (-1)**(am_i+am_j)*torch.transpose(sk_block, dim0=1, dim1=2)
                                    else:
                                        tmp_block[split_index][:,offset_i:offset_i+orbitnum_i,
                                                        offset_j:offset_j+orbitnum_j] = sk_block

                                    hopping_offset+=hoppingparamnum

                                    offset_j += orbitnum_j
                                offset_i += orbitnum_i
                            #tmp_block=torch.cat(tmp_block)
                        
                        dataset_name = f"type_{type_i}_{type_j}_split_{split_index}"
                        f.create_dataset(dataset_name, data=tmp_block[split_index].cpu().numpy()*27.211324570274, compression='gzip')
                    type_hop_offset+=1

    def find_realspace_hamiltonian_onsiteblock(self, data):
        
        atomtype=data[0].Atomtype.to(torch.int32).chunk(self.para.nn_split_scale,dim=0)

        file_path = os.path.join(self.para.dump_sparseH, 'onsites.h5')

        Des_radial_ons_1D=self.Des_radial_ons_1D.chunk(self.para.nn_split_scale,dim=0)

        with h5py.File(file_path, 'w') as f:

            for split_index in range(self.para.nn_split_scale):

                onsite_matrixes=0
                
                Des_radial_ons_1D_val = self.Des_to_Site_net(atomtype[split_index], Des_radial_ons_1D[split_index]).unsqueeze(-1)#.chunk(self.para.nn_split_scale,dim=0)
                
                for atom_type in range(self.para.num_types):

                    type_mask=atomtype[split_index]==atom_type
                    ons_1D=Des_radial_ons_1D_val[type_mask]
                    tmp_onsite_matrixes = torch.zeros(data.num_nodes//self.para.nn_split_scale,sum(torch.as_tensor(self.Index.orbit[atom_type])*2+1).item(),device=self.device)
                    offset = self.OnsiteEnergy_offset[atom_type]
                    orbitnum_am=0

                    for am_index, am in enumerate(self.Index.orbit[atom_type]):

                        oe = self.OnsiteEnergy[offset]

                        oe2=ons_1D[:,am_index]

                        orbitnum_am_plus = 2*am+1

                        tmp_onsite_matrixes[type_mask,orbitnum_am:orbitnum_am+orbitnum_am_plus]=(oe+oe2)

                        orbitnum_am += orbitnum_am_plus

                        offset+=1

                    onsite_matrixes+=tmp_onsite_matrixes#.flatten()

                dataset_name = f"split_{split_index}"
                f.create_dataset(dataset_name, data=onsite_matrixes.cpu().numpy()*27.211324570274, compression='gzip')

        
    def find_Rspace_hamiltonian(self, data,batch_index):

        self.find_realspace_hamiltonian_hoppingblock(data,batch_index)

        self.find_realspace_hamiltonian_onsiteblock(data)


