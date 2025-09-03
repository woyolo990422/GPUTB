import sys
import os
env_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(env_directory)
from time import time
from LSQT.Run import Run,Reform
from utilities.seed import set_seed
sys.dont_write_bytecode = True

def main(runin: dict):

    if runin.get("seed", None) != None:
        set_seed(runin["seed"])

    print("-"*50)
    print("Started executing the commands in run.in.")
    print("-"*50)

    time_begin = time()
    run = Run(runin)
    Reform(runin)
    time_finish = time()

    print("-"*50)
    print("Finished executing the commands in run.in.")
    print("-"*50)

    time_used = time_finish-time_begin
    print("-"*50)
    print(f"Time used = {time_used:.3f} s.")
    print("-"*50)
    

if __name__ == '__main__':
    from importlib.machinery import SourceFileLoader
    input_module = SourceFileLoader("input", "lsqt_input.py").load_module()
    runin = input_module.runin
    main(runin)

#if __name__ == '__main__':
#    from lsqt_input import runin
#    main(runin)