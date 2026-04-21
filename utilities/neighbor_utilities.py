import numpy as np
from ase.geometry import  complete_cell
import gc

def find_remain_index_2(i_, j_, S, symbols_numbers):

    i=symbols_numbers[i_]
    j=symbols_numbers[j_]

    mask = (i<j)|\
            ((i==j)&\
             ((S[:,0]<0)|\
              ((S[:,0] == 0) & (S[:,1] < 0))|\
                ((S[:,0] == 0) & (S[:,1] ==0)&(S[:,2] < 0))|\
                    ((i_<j_)&((S[:,0] == 0)&(S[:,1] == 0)&(S[:,2] == 0))))
                    )
    
    index=np.arange(i.shape[0])
    remain_index=index[mask]
    return remain_index

def primitive_neighbor_list(quantities, pbc, cell, positions, cutoff, numbers=None, self_interaction=False,
                            use_scaled_positions=False, max_nbins=1e6):

    cell=cell.astype(np.float32)
    positions=positions.astype(np.float32)

    if len(positions) == 0:
        empty_types = dict(i=(np.int32, (0, )),
                           j=(np.int32, (0, )),
                           D=(np.float32, (0, 3)),
                           d=(np.float32, (0, )),
                           S=(np.int32, (0, 3)))
        retvals = []
        for i in quantities:
            dtype, shape = empty_types[i]
            retvals += [np.array([], dtype=dtype).reshape(shape)]
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    # Compute reciprocal lattice vectors.
    b1_c, b2_c, b3_c = np.linalg.pinv(cell).T

    # Compute distances of cell faces.
    l1 = np.linalg.norm(b1_c)
    l2 = np.linalg.norm(b2_c)
    l3 = np.linalg.norm(b3_c)
    face_dist_c = np.array([1 / l1 if l1 > 0 else 1,
                            1 / l2 if l2 > 0 else 1,
                            1 / l3 if l3 > 0 else 1],dtype=np.float32)

    if isinstance(cutoff, dict):
        max_cutoff = max(cutoff.values())
    else:
        if np.isscalar(cutoff):
            max_cutoff = cutoff
        else:
            cutoff = np.asarray(cutoff)
            max_cutoff = 2 * np.max(cutoff)

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    nbins_c = np.maximum((face_dist_c / bin_size), [1, 1, 1]).astype(np.int32)
    nbins = np.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = np.maximum(nbins_c // 2, [1, 1, 1])
        nbins = np.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search_x, neigh_search_y, neigh_search_z = np.ceil(bin_size * nbins_c / face_dist_c).astype(np.int8)

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    neigh_search_x = 0 if nbins_c[0] == 1 and not pbc[0] else neigh_search_x
    neigh_search_y = 0 if nbins_c[1] == 1 and not pbc[1] else neigh_search_y
    neigh_search_z = 0 if nbins_c[2] == 1 and not pbc[2] else neigh_search_z

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = np.dot(scaled_positions_ic, cell).astype(np.float32)
    else:
        scaled_positions_ic = np.linalg.solve(complete_cell(cell).T, positions.T).T.astype(np.float32)

    bin_index_ic = np.floor(scaled_positions_ic * nbins_c).astype(np.int32)
    cell_shift_ic = np.zeros_like(bin_index_ic,dtype=np.int8)

    for c in range(3):
        if pbc[c]:
            # (Note: np.divmod does not exist in older numpies)
            cell_shift_ic[:, c], bin_index_ic[:, c] = divmod(bin_index_ic[:, c], nbins_c[c])
        else:
            bin_index_ic[:, c] = np.clip(bin_index_ic[:, c], 0, nbins_c[c] - 1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_tmp=(bin_index_ic[:, 1] + nbins_c[1] * bin_index_ic[:, 2].astype(np.int32))
    bin_index_i = (bin_index_ic[:, 0] +(nbins_c[0] * bin_tmp).astype(np.int32))

    del bin_index_ic,scaled_positions_ic
    gc.collect()

    atom_i = np.argsort(bin_index_i,kind="Mergesort").astype(np.int32)

    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_natoms_per_bin = np.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -np.ones([nbins, max_natoms_per_bin], dtype=np.int32)
    for i in range(max_natoms_per_bin):
        # Create a mask array that identifies the first atom of each bin.
        mask = np.append([True], bin_index_i[:-1] != bin_index_i[1:])
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = np.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0
    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    atom_pairs_pn = np.indices((max_natoms_per_bin, max_natoms_per_bin), dtype=np.int32)
    atom_pairs_pn = atom_pairs_pn.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []


    binz_xyz, biny_xyz, binx_xyz = np.meshgrid(np.arange(nbins_c[2],dtype=np.int32),
                                               np.arange(nbins_c[1],dtype=np.int32),
                                               np.arange(nbins_c[0],dtype=np.int32),
                                               indexing='ij')

    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]

    first_at_neightuple_n_save=np.empty(0,dtype=np.int32)
    secnd_at_neightuple_n_save=np.empty(0,dtype=np.int32)
    distance_vector_nc_save=np.empty(0,dtype=np.float32)
    abs_distance_vector_n_save=np.empty(0,dtype=np.float32)
    cell_shift_vector_n_save=np.empty((0,3),dtype=np.int8)

    for dz in range(-neigh_search_z, neigh_search_z + 1):
        for dy in range(-neigh_search_y, neigh_search_y + 1):
            for dx in range(-neigh_search_x, neigh_search_x + 1):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (neighbinx_xyz + nbins_c[0] * (neighbiny_xyz + nbins_c[1] * neighbinz_xyz).astype(np.int32)).ravel()


                # Second atom in pair.
                _secnd_at_neightuple_n = atoms_in_bin_ba[neighbin_b][:, atom_pairs_pn[1]]

                # Shift vectors.
                _cell_shift_vector_x_n =  np.resize(shiftx_xyz.reshape(-1, 1), (max_natoms_per_bin**2, shiftx_xyz.size)).T.astype(np.int8)
                _cell_shift_vector_y_n =  np.resize(shifty_xyz.reshape(-1, 1), (max_natoms_per_bin**2, shifty_xyz.size)).T.astype(np.int8)
                _cell_shift_vector_z_n =  np.resize(shiftz_xyz.reshape(-1, 1), (max_natoms_per_bin**2, shiftz_xyz.size)).T.astype(np.int8)

                mask = np.logical_and(_first_at_neightuple_n != -1,
                                      _secnd_at_neightuple_n != -1)
                if mask.sum() > 0:

                    first_at_neightuple_n=_first_at_neightuple_n[mask]
                    secnd_at_neightuple_n=_secnd_at_neightuple_n[mask]
                    cell_shift_vector_n_x=_cell_shift_vector_x_n[mask]
                    cell_shift_vector_n_y=_cell_shift_vector_y_n[mask]
                    cell_shift_vector_n_z=_cell_shift_vector_z_n[mask]

                cell_shift_vector_n=np.transpose([cell_shift_vector_n_x,cell_shift_vector_n_y,cell_shift_vector_n_z])
        
                cell_shift_vector_n += cell_shift_ic[first_at_neightuple_n] - cell_shift_ic[secnd_at_neightuple_n]

                if not self_interaction:
                    m = np.logical_not(np.logical_and(first_at_neightuple_n == secnd_at_neightuple_n, (cell_shift_vector_n == 0).all(axis=1)))
                    first_at_neightuple_n = first_at_neightuple_n[m]
                    secnd_at_neightuple_n = secnd_at_neightuple_n[m]
                    cell_shift_vector_n = cell_shift_vector_n[m]
        
                for c in range(3):
                    if not pbc[c]:
                        m = cell_shift_vector_n[:, c] == 0
                        first_at_neightuple_n = first_at_neightuple_n[m]
                        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
                        cell_shift_vector_n = cell_shift_vector_n[m]

                i = np.argsort(first_at_neightuple_n,kind="Mergesort").astype(np.int32)
                first_at_neightuple_n = first_at_neightuple_n[i]
                secnd_at_neightuple_n = secnd_at_neightuple_n[i]
                cell_shift_vector_n = cell_shift_vector_n[i]

                del i
                gc.collect()

                distance_vector_nc = positions[secnd_at_neightuple_n] - positions[first_at_neightuple_n] + cell_shift_vector_n.dot(cell)

                #abs_distance_vector_n = np.sqrt(np.sum(distance_vector_nc * distance_vector_nc, axis=1))
                abs_distance_vector_n = np.linalg.norm((distance_vector_nc), axis=1)

                # We have still created too many pairs. Only keep those with distance
                # smaller than max_cutoff.

                mask = abs_distance_vector_n < max_cutoff

                first_at_neightuple_n = first_at_neightuple_n[mask]

                secnd_at_neightuple_n = secnd_at_neightuple_n[mask]

                cell_shift_vector_n= cell_shift_vector_n[mask]

                distance_vector_nc = distance_vector_nc[mask]

                abs_distance_vector_n = abs_distance_vector_n[mask]

                del mask
                gc.collect()

                first_at_neightuple_n_save=np.append(first_at_neightuple_n_save,first_at_neightuple_n)
                secnd_at_neightuple_n_save=np.append(secnd_at_neightuple_n_save,secnd_at_neightuple_n)
                cell_shift_vector_n_save=np.append(cell_shift_vector_n_save,cell_shift_vector_n,axis=0)
                distance_vector_nc_save=np.append(distance_vector_nc_save,distance_vector_nc)
                abs_distance_vector_n_save=np.append(abs_distance_vector_n_save,abs_distance_vector_n)


    del cell_shift_ic,cell_shift_vector_x_n,cell_shift_vector_y_n,cell_shift_vector_z_n,first_at_neightuple_nn,secnd_at_neightuple_nn
    gc.collect()

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == 'i':
            retvals += [first_at_neightuple_n_save]
            del first_at_neightuple_n_save
            gc.collect()
        elif q == 'j':
            retvals += [secnd_at_neightuple_n_save]
            del secnd_at_neightuple_n_save
            gc.collect()
        elif q == 'D':
            retvals += [distance_vector_nc_save]
            del distance_vector_nc_save
            gc.collect()
        elif q == 'd':
            retvals += [abs_distance_vector_n_save]
            del abs_distance_vector_n_save
            gc.collect()
        elif q == 'S':
            retvals += [cell_shift_vector_n_save]
            del cell_shift_vector_n_save
            gc.collect()
        else:
            raise ValueError('Unsupported quantity specified.')
    
    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)

def neighbor_list(quantities, a, cutoff, self_interaction=False,
                  max_nbins=1e10):
    
    return primitive_neighbor_list(quantities, a.pbc,
                                   a.get_cell(complete=True),
                                   a.positions, cutoff, numbers=a.numbers,
                                   self_interaction=self_interaction,
                                   max_nbins=max_nbins)

