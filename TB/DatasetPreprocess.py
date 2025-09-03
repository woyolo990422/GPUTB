import numpy as np
import torch
from torch_geometric.data import Data
from ase.atom import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list as neighbor_list_train
import os
from TB.Parameters import Parameters
from utilities.neighbor_utilities import find_remain_index_2
from utilities.neighbor_utilities import neighbor_list as neighbor_list_H
from utilities.Index import Index
import h5py

class DatasetPrepocess(object):
    def __init__(self, para: Parameters, index: Index):
    
        if hasattr(para, 'compute_sparseH'):
            self.para = para
            self.dtype = torch.float32
            self.device = para.device
            self.cutoff = para.cutoff
            self.Index=index
            self.AtomSymbol_to_AtomNumber = {atomsymbol: atomnumber for atomsymbol, atomnumber in atomic_numbers.items() if atomsymbol in para.orbit.keys()}
            self.AtomNumber_to_AtomSymbol = {atomnumber: atomsymbol for atomsymbol, atomnumber in self.AtomSymbol_to_AtomNumber.items()}
            unique_type = sorted(self.AtomNumber_to_AtomSymbol.keys())
            self.AtomNumber_to_AtomType = {num: i for i, num in enumerate(unique_type)}
            self.AtomSymbol_to_AtomType = {self.AtomNumber_to_AtomSymbol[atomnumber]: atomtype for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}

            self.dataset = self.read_structures_sparse_H()
        else:
            self.para = para
            self.dataset_list = para.dataset
            self.dtype = torch.float32
            self.device = para.device
            self.cutoff = para.cutoff
            self.Index=index
            self.AtomSymbol_to_AtomNumber = {atomsymbol: atomnumber for atomsymbol, atomnumber in atomic_numbers.items() if atomsymbol in para.orbit.keys()}
            self.AtomNumber_to_AtomSymbol = {atomnumber: atomsymbol for atomsymbol, atomnumber in self.AtomSymbol_to_AtomNumber.items()}
            unique_type = sorted(self.AtomNumber_to_AtomSymbol.keys())
            self.AtomNumber_to_AtomType = {num: i for i, num in enumerate(unique_type)}
            self.AtomSymbol_to_AtomType = {self.AtomNumber_to_AtomSymbol[atomnumber]: atomtype for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
            self.AtomType_to_AtomNumber = {atomtype: atomnumber for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
            self.AtomType_to_AtomSymbol = {atomtype: self.AtomNumber_to_AtomSymbol[atomnumber] for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
            self.type_onehot = torch.eye(len(self.AtomSymbol_to_AtomType))
            self.embeding = torch.nn.Embedding(len(self.AtomSymbol_to_AtomType), 16)
            self.dataset = self.read_structures()

    def find_orbit_information(self,num_nodes,Atomtype):
        tmp_total_orbit_num = 0
        orbit_num_per_atom = np.zeros(num_nodes, dtype=int)
        orbit_offset = np.zeros(num_nodes, dtype=int)
        for atom_index, atomtype in enumerate(Atomtype):
            orbit_num = 0
            orbit_num=torch.sum(torch.as_tensor(self.Index.orbit[atomtype.item()],dtype=torch.int32)*2+1)
            orbit_num_per_atom[atom_index] = orbit_num
            orbit_offset[atom_index] = tmp_total_orbit_num
            tmp_total_orbit_num += orbit_num
        return tmp_total_orbit_num, torch.from_numpy(orbit_num_per_atom), torch.from_numpy(orbit_offset)
    
    def find_orbit_information_train(self,num_nodes,Atomtype):
        tmp_total_orbit_num = 0
        orbit_num_per_atom = np.zeros(num_nodes, dtype=int)
        orbit_offset = np.zeros(num_nodes, dtype=int)
        for atom_index, atomtype in enumerate(Atomtype):
            orbit_num = 0
            orbit_num=torch.sum(torch.as_tensor(self.Index.orbit[atomtype.item()],dtype=torch.int32)*2+1)
            orbit_num_per_atom[atom_index] = orbit_num
            orbit_offset[atom_index] = tmp_total_orbit_num
            tmp_total_orbit_num += orbit_num
           

        return tmp_total_orbit_num, orbit_num_per_atom.tolist(), orbit_offset.tolist()


    def find_split_num(self,a, b, target):
        gcd_ab = torch.gcd(torch.tensor(a,dtype=torch.int64), torch.tensor(b,dtype=torch.int64))
        divisors = []
        for i in range(1, gcd_ab.item() + 1):
            if a % i == 0 and b % i == 0:
                divisors.append(i)
        divisors_tensor = torch.tensor(divisors)
        target_tensor = torch.tensor(target)
        closest_idx = torch.argmin(torch.abs(divisors_tensor - target_tensor))
        closest = divisors_tensor[closest_idx].item()
        return closest
    
    def sort_type_hop(self,Atomtype,i_hop,j_hop):

        resort_hop_=torch.empty(0,dtype=torch.int32,device=self.device)
        resort_hop_index_=[0]
        length_offset_hop=0
        for type_i in range(self.para.num_types):
            for type_j in range(type_i,self.para.num_types):
                hop_index_seleced = torch.arange(len(i_hop),dtype=torch.int32, device=self.device)[((Atomtype[i_hop] == type_i)*(Atomtype[j_hop] == type_j))]
                resort_hop_=torch.cat((resort_hop_,hop_index_seleced))
                length_offset_hop+=len(hop_index_seleced)
                resort_hop_index_.append(length_offset_hop)
        return resort_hop_index_,resort_hop_

    def read_structures_sparse_H(self):
        dataset=[]

        structures = read(self.para.structure_path)

        Atomtype = torch.as_tensor([self.AtomSymbol_to_AtomType[symbol] for symbol in structures.symbols], dtype=torch.int8)

        lattice = torch.as_tensor(structures.cell.array).unsqueeze(0).to(torch.float32)
        
        pos = torch.from_numpy(structures.positions).to(torch.float32)

        if max(Atomtype)<2:
            i_ons, j_ons, d_ons, D_ons, S_ons = neighbor_list_H("ijdDS",a=structures,cutoff=self.para.cutoff)
        else:
            i_ons, j_ons, d_ons, D_ons, S_ons = neighbor_list_train("ijdDS",a=structures,cutoff=self.para.cutoff)


        D_ons=D_ons.reshape(-1,3)

        self.para.nn_split_scale=self.find_split_num(i_ons.shape[0]//2,Atomtype.shape[0],self.para.nn_split_scale)

        index_remain = torch.as_tensor(find_remain_index_2(i_ons, j_ons, S_ons,structures.symbols.numbers),dtype=torch.int32).chunk(self.para.nn_split_scale)
        #find_smart scale

        i_ons=torch.as_tensor(i_ons,dtype=torch.int32,device=self.device)
        j_ons=torch.as_tensor(j_ons,dtype=torch.int32,device=self.device)
        d_ons=torch.as_tensor(d_ons,dtype=torch.float32,device=self.device)
        D_ons=torch.as_tensor(D_ons,dtype=torch.float32,device=self.device)

        total_orbit_num, orbit_num_per_atom, orbit_offset = self.find_orbit_information(Atomtype.shape[0],Atomtype)

        i_hop=torch.empty(0,dtype=torch.int32,device=self.device)
        j_hop=torch.empty(0,dtype=torch.int32,device=self.device)
        d_hop=torch.empty(0,dtype=torch.float32,device=self.device)
        D_hop=torch.empty((0,3),dtype=torch.float32,device=self.device)


        for split_index in range(len(index_remain)):
            tmp_remain=index_remain[split_index]
            tmp_i_hop=i_ons[tmp_remain]
            tmp_j_hop=j_ons[tmp_remain]
            tmp_d_hop=d_ons[tmp_remain]
            tmp_D_hop=D_ons[tmp_remain]
            i_hop=torch.cat((tmp_i_hop,i_hop))
            j_hop=torch.cat((tmp_j_hop,j_hop))
            d_hop=torch.cat((tmp_d_hop,d_hop))
            D_hop=torch.cat((tmp_D_hop,D_hop),dim=0)


        resort_hop_index_,resort_hop_=self.sort_type_hop(Atomtype,i_hop,j_hop)

        i_hop_=i_hop[resort_hop_]
        j_hop_=j_hop[resort_hop_]
        d_hop_=d_hop[resort_hop_]
        D_hop_=D_hop[resort_hop_]
        
        or_li=orbit_num_per_atom[i_hop_]
        or_lj=orbit_num_per_atom[j_hop_]

        block_offset=torch.cumsum(or_li*or_lj, dim=0)

        data = Data(Atomtype=Atomtype,
                    lattice=lattice,
                    pos=pos,
                    pbc=structures.pbc,
                    index_remain=index_remain,
                    structures_volume=structures.get_volume(),
                    num_edge=i_ons.shape[0]//2,
                    i_hop=i_hop_,
                    j_hop=j_hop_,
                    d_hop=d_hop_,
                    D_hop=D_hop_,                    
                    i_ons=i_ons,
                    j_ons=j_ons,
                    d_ons=d_ons,
                    D_ons=D_ons, 
                    total_orbit_num=total_orbit_num,
                    orbit_num_per_atom=orbit_num_per_atom,
                    orbit_offset=orbit_offset,
                    resort_hop_index=resort_hop_index_
                    )
        file_path = os.path.join(self.para.dump_sparseH, 'model.h5')

        orbit_length=[]

        for i in range(len(self.Index.orbit)):
            tmp_i=self.Index.orbit[i]
            orbit_length.append(sum(torch.as_tensor(tmp_i)*2+1))


        with h5py.File(file_path, 'w') as f:

            f.create_dataset('i_hop', data=i_hop_)
            f.create_dataset('j_hop', data=j_hop_)
            f.create_dataset('D_hop', data=D_hop_[:,self.para.transport_direction-1])
            f.create_dataset('block_offset', data=block_offset)
            f.create_dataset('orbit_length', data=orbit_length)
            f.create_dataset('num_types', data=self.para.num_types)
            f.create_dataset('time_step', data=self.para.time_step)
            f.create_dataset('max_energy', data=self.para.max_energy)
            f.create_dataset('start_energy', data=self.para.start_energy)
            f.create_dataset('end_energy', data=self.para.end_energy)
            f.create_dataset('num_moments', data=self.para.num_moments)           
            f.create_dataset('num_energies', data=self.para.num_energies)           
            f.create_dataset('transport_direction', data=self.para.transport_direction)  
            f.create_dataset('max_orbit_neighbor', data=self.para.compute_sparseH[1])  
            f.create_dataset('num_of_atoms', data=structures.numbers.shape[0])           
            f.create_dataset('num_of_steps', data=self.para.num_of_steps)           
            f.create_dataset('volume', data=structures.get_volume())   
            f.create_dataset('orbit_num_per_atom', data=orbit_num_per_atom)           
            f.create_dataset('orbit_offset', data=orbit_offset)           
            f.create_dataset('nn_split_scale', data=self.para.nn_split_scale)
            f.create_dataset('lsqt_split_scale', data=self.para.lsqt_split_scale)
            f.create_dataset('num_of_orbits', data=total_orbit_num)    
            f.create_dataset('num_of_edges', data=i_ons.shape[0]//2)           
            f.create_dataset('resort_hop_index', data=resort_hop_index_)
            f.create_dataset('max_orbit', data=orbit_num_per_atom.max())
            pass
            # if  self.para.classical_sk:            
            #     f.create_dataset('x_pos', data=pos[:,0])
            #     f.create_dataset('y_pos', data=pos[:,1])
            #     f.create_dataset('z_pos', data=pos[:,2])
            #     f.create_dataset('lattice', data=data.lattice[0])
            #     f.create_dataset('pbc', data=data.pbc)
            #     f.create_dataset('number_of_orbitals_per_atom', data=1)
              
        dataset.append(data.to(device=self.device))

        return dataset

    def read_structures(self):
        dataset = []
        dataset_id=0
        for dirname in self.para.dataset:
            kpoints_file = os.path.join(dirname, "kpoints.npy")
            eigs_file = os.path.join(dirname, "eigs.npy")
            structure_file = os.path.join(dirname, "xdat.traj")

            kpoints = torch.from_numpy(np.load(kpoints_file)).to(torch.float32)[::self.para.plot_gap]
            eigs_ref = torch.from_numpy(np.load(eigs_file)).to(torch.float32)

            if len(eigs_ref.shape) == 2:
                eigs_ref = eigs_ref[::self.para.plot_gap]
            elif len(eigs_ref.shape) == 3:
                eigs_ref = eigs_ref[:,::self.para.plot_gap,:]

            structures = read(structure_file, index=":")
            if len(eigs_ref.shape)==2:
                eigs_ref=eigs_ref.reshape(1,eigs_ref.shape[0],eigs_ref.shape[1])

            assert len(kpoints.shape) == 2 and len(eigs_ref.shape) == 3


            for frame_index, frame in enumerate(structures):

                Atomtype = torch.tensor([self.AtomSymbol_to_AtomType[symbol] for symbol in frame.symbols], dtype=torch.int32)
                x = self.embeding(Atomtype)
                lattice = torch.tensor(frame.cell.array).unsqueeze(0).to(torch.float32)
                pos = torch.from_numpy(frame.positions).to(torch.float32)

                i_hop, j_hop, d_hop, D_hop, S_hop = neighbor_list_train("ijdDS", a=frame,cutoff=self.para.cutoff)

                i_ons, j_ons, d_ons, D_ons, S_ons=i_hop, j_hop, d_hop, D_hop, S_hop
                
                index_remain = find_remain_index_2(i_hop, j_hop, S_hop, frame.symbols.numbers)

                i_hop = i_hop[index_remain]
                j_hop = j_hop[index_remain]
                d_hop = d_hop[index_remain]
                D_hop = D_hop[index_remain]
                S_hop = S_hop[index_remain]

                edge_index_hop = torch.stack([torch.LongTensor(i_hop), torch.LongTensor(j_hop)], dim=0)
                
                total_orbit_num, orbit_num_per_atom, orbit_offset = self.find_orbit_information(Atomtype.shape[0],Atomtype)

                data = Data(x=x,
                            Atomtype=Atomtype,
                            lattice=lattice,
                            pos=pos,
                            kpoints=kpoints,#.reshape(-1),#.unsqueeze(0),
                            frame_offset=frame_index*torch.ones(edge_index_hop.shape[1],dtype=int),
                            frame_symbols=frame.symbols,
                            eigs_ref=eigs_ref[frame_index].unsqueeze(0),#.reshape(-1),
                            edge_index_hop=edge_index_hop,
                            num_edge=edge_index_hop.shape[1],
                            d_hop=torch.from_numpy(d_hop).to(torch.float32),
                            D_hop=torch.from_numpy(D_hop).to(torch.float32),
                            S_hop=torch.from_numpy(S_hop).to(torch.float32),
                            total_orbit_num=total_orbit_num,
                            orbit_num_per_atom=orbit_num_per_atom,
                            orbit_offset=orbit_offset,
                            dataset_id=torch.as_tensor(dataset_id,dtype=torch.int32).reshape(-1),
                            )
                


                edge_index_ons = torch.stack([torch.LongTensor(i_ons), torch.LongTensor(j_ons)], dim=0)
                data.edge_index_ons = edge_index_ons
                data.d_ons = torch.from_numpy(d_ons).to(torch.float32)
                data.D_ons = torch.from_numpy(D_ons).to(torch.float32)
                data.S_ons = torch.from_numpy(S_ons).to(torch.int32)

                dataset.append(data.to(device=self.device))


            dataset_id+=1

        return dataset

