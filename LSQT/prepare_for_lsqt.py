import torch

def find_closest_divisor(a, b):
    c = b
    while c >= 1:
        if a % c == 0:
            return c
        c -= 1
    return None


class For_lsqt(object):
    #@profile
    def __init__(self, tbmodel,para, input):
        self.para=para
        self.tbmodel = tbmodel
        self.input = input
        self.tbmodel.cutoff = self.input["cutoff"]
        self.tbmodel.load_state_dict(torch.load(self.input["model_init_path"], map_location=torch.device('cpu')), strict=False)
        self.GPUMD_format = {}
        self.dataloader=self.tbmodel.dataloader

    #@profile
    def compute(self):
        self.tbmodel.eval()
        with torch.no_grad():
            for data in self.dataloader:
                data.to(self.tbmodel.device)
                self.tbmodel.find_Rspace_hamiltonian(data,0)
            

            pass
