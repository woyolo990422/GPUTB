from time import time
import sys
import os
env_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(env_directory)
from TB.Parameters import Parameters
from TB.Fitness import Fitness
from TB.Train import Train
from utilities.Index import Index
from utilities.seed import set_seed
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.")
sys.dont_write_bytecode = True


def main(nepin: dict):
    if nepin.get("seed", None) != None:
        set_seed(nepin["seed"])
        torch.cuda.manual_seed(nepin["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    time_begin = time()
    para = Parameters(nepin)
    index = Index(para.nepin["orbit"])
    fitness = Fitness(para, index)
    time_finish = time()

    time_used = time_finish-time_begin
    print("-"*50)
    print(f"Time used for initialization = {time_used:.3f} s.")
    print("-"*50)

    time_begin = time()
    Train(para, fitness, index)
    time_finish = time()

    time_used = time_finish-time_begin
    print("-"*50)
    if (para.nepin["prediction"] == 0):
        print(f"Time used for training = {time_used:.3f} s.")
    elif (para.nepin["prediction"] == 1):
        print(f"Time used for predicting = {time_used:.3f} s.")
    print("-"*50)

#if __name__ == '__main__':
    #from Si.Si import nepin as trainin
    #from deeptb.AlAs.AlAs import nepin as trainin
    #from GaP.GaP import nepin as trainin
    #from deeptb.GaP.GaP import nepin as trainin
    #from deeptb.InAs.InAs import nepin as trainin
    #from AlGaP2.AlGaP2 import nepin as trainin
    #from Ag.Ag import nepin as trainin
    #from SiGe.SiGe import nepin as  trainin
    #from SnSe.SnSe import nepin as trainin

    #from deeptb.C.C import nepin as trainin
    #from BN.BN import nepin as trainin

    #from deeptb.InAs.InAs import nepin as trainin
    #main(trainin)
 
if __name__ == '__main__':
    from importlib.machinery import SourceFileLoader

    input_module = SourceFileLoader("input", "input.py").load_module()
    nepin = input_module.nepin
    main(nepin)
