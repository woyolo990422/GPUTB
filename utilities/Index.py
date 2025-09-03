import numpy as np
import torch
import re
import ase


class Index(object):
    Symbol_to_AtomNumber = ase.atom.atomic_numbers
    AngularMomentum = {'s': 0, 'p': 1, 'd': 2, 'f': 3}

    def __init__(self, orbit_input: dict) -> None:
        self.orbit_input = orbit_input
        self.symbols = self.orbit_input.keys()

        self.Symbol_to_AtomType = self.find_Symbol_to_AtomType()
        self.orbit = self.find_orbit()

        self.HoppingParamNum = self.find_HoppingParamNum()
        self.HoppingIndex_ptr = self.find_HoppingIndex_ptr()
        self.Hopping_ptr = self.find_hopping_ptr()

        self.Onsitedeform_correctParamNum = self.find_Onsitedeform_correctParamNum()
        self.Onsitedeform_correctIndex_ptr = self.find_Onsitedeform_correctIndex_ptr()

    def find_Symbol_to_AtomType(self):
        AN = [self.Symbol_to_AtomNumber[symbol]
              for symbol in self.symbols]
        AN_sorted = sorted(AN)
        index_sorted = [AN_sorted.index(an) for an in AN]
        Symbol_to_AtomType = dict(zip(self.symbols, index_sorted))
        return Symbol_to_AtomType

    def find_orbit(self):
        orbit = {}
        for symbol in self.orbit_input:
            AMs = []
            for AM in self.orbit_input[symbol]:
                AM = "".join(re.findall(r"[a-z]", AM))
                AMs.append(self.AngularMomentum[AM])
            atomtype = self.Symbol_to_AtomType[symbol]
            orbit[atomtype] = AMs
        return dict(sorted(orbit.items()))  # 原子序数小的排前面

    def find_HoppingParamNum(self):
        HoppingParamNum = 0
        for atomtype_i in self.orbit:
            for atomtype_j in self.orbit:
                for am_i_index, am_i in enumerate(self.orbit[atomtype_i]):
                    for am_j_index, am_j in enumerate(self.orbit[atomtype_j]):

                        if (atomtype_i == atomtype_j):
                            # if am_i_index <= am_j_index:
                                HoppingParamNum += min(am_i, am_j)+1

                        elif (atomtype_i < atomtype_j):
                            HoppingParamNum += min(am_i, am_j)+1
        return HoppingParamNum

    def find_HoppingIndex_ptr(self):
        hoppingindex_ptr = [[0 for j in range(len(self.orbit))]
                            for i in range(len(self.orbit))]

        tmp_index = 0
        for atomtype_i in self.orbit:
            for atomtype_j in self.orbit:

                if hoppingindex_ptr[atomtype_j][atomtype_i] != 0:
                    hoppingindex_ptr[atomtype_i][atomtype_j] = hoppingindex_ptr[atomtype_j][atomtype_i]

                else:
                    HoppingParamNum = 0
                    for am_i_index, am_i in enumerate(self.orbit[atomtype_i]):
                        for am_j_index, am_j in enumerate(self.orbit[atomtype_j]):

                            # if (atomtype_i == atomtype_j):
                            #     if am_i_index <= am_j_index:
                            #         HoppingParamNum += min(am_i, am_j)+1

                            # elif (atomtype_i < atomtype_j):
                                HoppingParamNum += min(am_i, am_j)+1
                    hoppingindex_ptr[atomtype_i][atomtype_j] = [
                        tmp_index, tmp_index+HoppingParamNum]
                    tmp_index += HoppingParamNum
        return hoppingindex_ptr

    def find_hopping_ptr(self):
        hopping_ptr = [[0 for j in range(len(self.symbols))]
                       for i in range(len(self.symbols))]
        random_alpha1 = np.random.uniform(-1, 1, (self.HoppingParamNum))
        random_alpha2 = np.random.uniform(-1, 1, (self.HoppingParamNum))

        for atomtype_i in self.orbit:
            for atomtype_j in self.orbit:
                index_start, index_end = self.HoppingIndex_ptr[atomtype_i][atomtype_j]
                ParaNum = index_end-index_start
                hopping_ptr[atomtype_i][atomtype_j] = [
                    ParaNum, random_alpha1[index_start:index_end], random_alpha2[index_start:index_end]]

        return hopping_ptr

    def find_Onsitedeform_correctParamNum(self):
        Onsitedeform_correctParamNum = 0
        for atomtype_i in self.orbit:
            for atomtype_j in self.orbit:
                for am_i_index, am_i in enumerate(self.orbit[atomtype_i]):
                    for am_j_index, am_j in enumerate(self.orbit[atomtype_i]):

                        if (am_i_index <= am_j_index):
                            Onsitedeform_correctParamNum += min(am_i, am_j)+1

        return Onsitedeform_correctParamNum

    def find_Onsitedeform_correctIndex_ptr(self):
        Onsitedeform_correctIndex_ptr = [[0 for j in range(len(self.orbit))]
                                 for i in range(len(self.orbit))]

        tmp_index = 0
        for atomtype_i in self.orbit:
            for atomtype_j in self.orbit:

                Onsitedeform_correctParamNum = 0
                for am_i_index, am_i in enumerate(self.orbit[atomtype_i]):
                    for am_j_index, am_j in enumerate(self.orbit[atomtype_i]):

                        if (am_i_index <= am_j_index):
                            Onsitedeform_correctParamNum += min(am_i, am_j)+1

                Onsitedeform_correctIndex_ptr[atomtype_i][atomtype_j] = [
                    tmp_index, tmp_index+Onsitedeform_correctParamNum]
                tmp_index += Onsitedeform_correctParamNum
        return Onsitedeform_correctIndex_ptr
