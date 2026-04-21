import torch
import ase

class Parameters(object):
    def __init__(self, nepin: dict):
        self.nepin = {}
        self.set_default_parameters()
        self.read_nep_in(nepin)
        self.__dict__.update(self.nepin)

        self.atomic_numbers = [ase.atom.atomic_numbers[ele] for ele in self.orbit.keys()]
        self.num_types=len(self.atomic_numbers)

    def set_default_parameters(self):
        default_dict = {}
        default_dict["dtype"] = torch.float32,
        default_dict["device"] = "cpu"
        default_dict["model_init_path"] = None
        self.nepin.update(default_dict)

    def read_nep_in(self, nepin):
        self.nepin.update(nepin)


