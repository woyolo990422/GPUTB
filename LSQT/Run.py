import os
import numpy as np
import h5py
from LSQT.prepare_for_lsqt import For_lsqt
from TB.train_model import Train_Model as T_model
from LSQT.lsqt_model import Train_Model as L_model
from utilities.Index import Index
from TB.Parameters import Parameters
from TB.DatasetPreprocess import DatasetPrepocess
from torch_geometric.loader import DataLoader
from numba import njit

class Run(object):
    def __init__(self, runin: dict):
        self.runin = {}
        self.Index = Index(runin["orbit"])
        self.read_run_in(runin)
        self.para=Parameters(self.runin)
        self.parse_run_in()

    def read_run_in(self, runin):
        self.runin.update(runin)

    def parse_run_in(self):
        keys = self.runin.keys()
        self.data = DatasetPrepocess(self.para, self.Index).dataset
        self.dataloader = DataLoader(self.data)

        if "compute_sparseH" in keys:
            self.tbmodel = L_model(self.para, self.Index,self.dataloader)
            self.lsqt=For_lsqt(self.tbmodel,self.para, self.runin)
            self.lsqt.compute()
        else:
            self.tbmodel = T_model(self.para, self.Index,self.dataloader)

class Reform(object):
    def __init__(self, runin: dict):
        self.runin = {}
        self.read_run_in(runin)
        self.para=Parameters(self.runin)
        if self.para.lsqt_split_scale<=1:
            self.direct_run()
        else:
            self.split_run()


    def direct_run(self):
        model_path = self.para.dump_sparseH+"/"+"model.h5"
        with h5py.File(model_path, 'r') as f:
            i_hop = np.array(f['i_hop'][:],dtype=np.int32)
            j_hop = np.array(f['j_hop'][:],dtype=np.int32)
            D_hop = np.array(f['D_hop'][:],dtype=np.float32)
            resort_hop_index = np.array(f['resort_hop_index'][:],dtype=np.int32)
            orbit_length = f['orbit_length'][:]
            num_types = f['num_types'][()]
            max_orbit_neighbor = f['max_orbit_neighbor'][()]
            orbit_offset = f['orbit_offset'][:]
            num_of_atoms = f['num_of_atoms'][()]
            num_of_orbits = f['num_of_orbits'][()]

        hoppings_path = self.para.dump_sparseH+"/"+"hoppings.h5"
        tmp_block=[]
        with h5py.File(hoppings_path, 'r') as f:
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                tmp_block.append(dataset[:])
        hoppings_block = np.concatenate(tmp_block, axis=0)

        onsites_path = self.para.dump_sparseH+"/"+"onsites.h5"
        tmp_block=[]
        with h5py.File(onsites_path, 'r') as f:
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                tmp_block.append(dataset[:])
        onsites_block = np.concatenate(tmp_block, axis=0)

        shift_i,shift_j=hash_f(i_hop,j_hop)

        Hr_hopping_val = np.zeros(max_orbit_neighbor * num_of_orbits, dtype=np.float32)
        #NN_orbit = np.ones(num_of_atoms, dtype=np.int32) * max_orbit_neighbor
        NL_orbit = -1*np.ones(max_orbit_neighbor * num_of_atoms, dtype=np.int32)
        xx = np.zeros(max_orbit_neighbor * num_of_atoms, dtype=np.float32)

        max_l = orbit_length.max()

        bond_type=0

        for type_i in range(num_types):
            for type_j in range(type_i, num_types):
                oi_l = orbit_length[type_i]
                oj_l = orbit_length[type_j]

                resort_s = resort_hop_index[bond_type]
                resort_e = resort_hop_index[bond_type + 1]
                bond_type+=1
                hoppings_sliced = hoppings_block[resort_s:resort_e]
                i_hop_sliced = i_hop[resort_s:resort_e]
                j_hop_sliced = j_hop[resort_s:resort_e]
                shift_i_sliced = shift_i[resort_s:resort_e]
                shift_j_sliced = shift_j[resort_s:resort_e]
                D_hop_sliced = D_hop[resort_s:resort_e]

                offset_i = orbit_offset[i_hop_sliced]
                offset_j = orbit_offset[j_hop_sliced]

                for o_j in range(oj_l):
                    atom_index_i = (shift_i_sliced * max_l + o_j) * num_of_atoms  +i_hop_sliced
                    np.put(NL_orbit, atom_index_i, o_j + offset_j)
                    np.put(xx, atom_index_i, D_hop_sliced)

                for o_i in range(oi_l):
                    atom_index_j = (shift_j_sliced * max_l + o_i) * num_of_atoms  +j_hop_sliced
                    np.put(NL_orbit, atom_index_j, o_i + offset_i)
                    np.put(xx, atom_index_j, -D_hop_sliced)


                for o_i in range(oi_l):
                    for o_j in range(oj_l):

                        index_i = (shift_i_sliced * max_l + o_j) * num_of_orbits + o_i + offset_i
                        index_j = (shift_j_sliced * max_l + o_i) * num_of_orbits + o_j + offset_j
                        
                        single_val_i = hoppings_sliced[:, o_i, o_j]
                        
                        np.put(Hr_hopping_val, index_i, single_val_i)
                        np.put(Hr_hopping_val, index_j, single_val_i)
                                            

                pass
        NL_orbit_2D=NL_orbit.reshape(max_orbit_neighbor,num_of_atoms).T
        NN_orbit=np.sum(NL_orbit_2D!= -1, axis=1).max()

        with h5py.File(model_path, 'a') as f:
            del f['orbit_offset']
            del f['orbit_num_per_atom']
            del f['resort_hop_index']
            del f['orbit_length']
            del f['i_hop']
            del f['j_hop']
            del f['D_hop']
            del f['block_offset']

            f.create_dataset('Hr_hopping_val', data=Hr_hopping_val) #)
            f.create_dataset('Hr_onsite_val', data=onsites_block.ravel())
            f.create_dataset('NL_orbit', data=NL_orbit)      
            f.create_dataset('NN_orbit', data=NN_orbit)
            f.create_dataset('xx', data=xx)
            pass

        os.remove(onsites_path)
        os.remove(hoppings_path)

    def split_run(self):
        model_path = self.para.dump_sparseH+"/"+"model.h5"
        with h5py.File(model_path, 'r') as f:
            i_hop = np.array(f['i_hop'][:],dtype=np.int32)
            j_hop = np.array(f['j_hop'][:],dtype=np.int32)
            D_hop = np.array(f['D_hop'][:],dtype=np.float32)
            resort_hop_index = np.array(f['resort_hop_index'][:],dtype=np.int32)
            orbit_length = f['orbit_length'][:]
            num_types = f['num_types'][()]
            max_orbit_neighbor = f['max_orbit_neighbor'][()]
            orbit_offset = f['orbit_offset'][:]
            lsqt_split_scale = f['lsqt_split_scale'][()]
            num_of_atoms = f['num_of_atoms'][()]
            num_of_orbits = f['num_of_orbits'][()]

        hoppings_path = self.para.dump_sparseH+"/"+"hoppings.h5"
        tmp_block=[]
        with h5py.File(hoppings_path, 'r') as f:
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                tmp_block.append(dataset[:])
        hoppings_block = np.concatenate(tmp_block, axis=0)

        onsites_path = self.para.dump_sparseH+"/"+"onsites.h5"
        tmp_block=[]
        with h5py.File(onsites_path, 'r') as f:
            for dataset_name in f.keys():
                dataset = f[dataset_name]
                tmp_block.append(dataset[:])
        onsites_block = np.concatenate(tmp_block, axis=0)

        shift_i,shift_j=hash_f(i_hop,j_hop)

        Hr_hopping_val = np.zeros(max_orbit_neighbor * num_of_orbits, dtype=np.float32)
        #NN_orbit = np.ones(num_of_atoms, dtype=np.int32) * max_orbit_neighbor
        NL_orbit = np.zeros(max_orbit_neighbor * num_of_atoms, dtype=np.int32)
        xx = np.zeros(max_orbit_neighbor * num_of_atoms, dtype=np.float32)

        max_l = orbit_length.max()

        for type_i in range(num_types):
            for type_j in range(type_i, num_types):
                oi_l = orbit_length[type_i]
                oj_l = orbit_length[type_j]

                bond_type = type_i * type_i + type_j * type_j

                resort_s = resort_hop_index[bond_type]
                resort_e = resort_hop_index[bond_type + 1]

                split_array = np.linspace(resort_s, resort_e, dtype=np.int64, num=lsqt_split_scale)

                for split_index in range(lsqt_split_scale - 1):
                    split_s = split_array[split_index]
                    split_e = split_array[split_index + 1]

                    hoppings_sliced = hoppings_block[split_s:split_e]
                    i_hop_sliced = i_hop[split_s:split_e]
                    j_hop_sliced = j_hop[split_s:split_e]
                    shift_i_sliced = shift_i[split_s:split_e]
                    shift_j_sliced = shift_j[split_s:split_e]
                    D_hop_sliced = D_hop[split_s:split_e]

                    offset_i = orbit_offset[i_hop_sliced]
                    offset_j = orbit_offset[j_hop_sliced]

                    for o_j in range(oj_l):
                        atom_index_i = (shift_i_sliced * max_l + o_j) * num_of_atoms  +i_hop_sliced
                        np.put(NL_orbit, atom_index_i, o_j + offset_j)
                        np.put(xx, atom_index_i, D_hop_sliced)

                    for o_i in range(oi_l):
                        atom_index_j = (shift_j_sliced * max_l + o_i) * num_of_atoms  +j_hop_sliced
                        np.put(NL_orbit, atom_index_j, o_i + offset_i)
                        np.put(xx, atom_index_j, -D_hop_sliced)


                    for o_i in range(oi_l):
                        for o_j in range(oj_l):

                            index_i = (shift_i_sliced * max_l + o_j) * num_of_orbits + o_i + offset_i
                            index_j = (shift_j_sliced * max_l + o_i) * num_of_orbits + o_j + offset_j
                            
                            single_val_i = hoppings_sliced[:, o_i, o_j]
                            
                            np.put(Hr_hopping_val, index_i, single_val_i)
                            np.put(Hr_hopping_val, index_j, single_val_i)
                                                

                    pass
        NL_orbit_2D=NL_orbit.reshape(num_of_atoms,max_orbit_neighbor)
        NN_orbit=np.sum(NL_orbit_2D!= -1, axis=1)
        with h5py.File(model_path, 'a') as f:
            del f['orbit_offset']
            del f['orbit_num_per_atom']
            del f['resort_hop_index']
            del f['orbit_length']
            del f['i_hop']
            del f['j_hop']
            del f['D_hop']
            del f['block_offset']
            f.create_dataset('Hr_hopping_val', data=Hr_hopping_val)
            f.create_dataset('Hr_onsite_val', data=onsites_block.ravel())
            f.create_dataset('NL_orbit', data=NL_orbit)      
            f.create_dataset('NN_orbit', data=NN_orbit)
            f.create_dataset('xx', data=xx)

        os.remove(onsites_path)
        os.remove(hoppings_path)
    
    def read_run_in(self, runin):
        self.runin.update(runin)

@njit
def hash_f(arr1, arr2):
    hash_table = {}
    counts1 = np.zeros_like(arr1)
    counts2 = np.zeros_like(arr2)
    
    for i in range(len(arr1)):
        if arr1[i] in hash_table:
            counts1[i] = hash_table[arr1[i]] + 1
            hash_table[arr1[i]] += 1
        else:
            counts1[i] = 1
            hash_table[arr1[i]] = 1
        
        if arr2[i] in hash_table:
            counts2[i] = hash_table[arr2[i]] + 1
            hash_table[arr2[i]] += 1
        else:
            counts2[i] = 1
            hash_table[arr2[i]] = 1
    
    return counts1-1, counts2-1
