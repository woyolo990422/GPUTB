import torch
import os
from Parameters import Parameters
from Fitness import Fitness
from utilities.Index import Index
import time

class Train(torch.nn.Module):
    def __init__(self, para: Parameters, fitness: Fitness, index: Index):
        self.para = para
        self.fitness = fitness
        self.index = index
        self.maximum_generation =self.para.nepin["generation"] 
        
        self.initialize_model()
        self.compute()

    def compute(self):
        start_time = time.time()

        if (self.para.nepin["prediction"] == 0):
            print(f"generation  MSE-EIG-Train MSE-EIG-Test MAE-EIG-Test lr elapsed_time")
            for generation in range(self.maximum_generation):
                elapsed_time = time.time() - start_time
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = elapsed_time % 60
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:05.3f}"

                self.fitness.compute(generation,time_str)


                if ((generation+1) % self.para.nepin["step_interval"] == 0):
                    self.output_model()

                if ((generation+1) % 1000 == 0):
                    self.output_check((generation+1))

        elif (self.para.nepin["prediction"] == 1):
            self.fitness.predict()
        elif (self.para.nepin["prediction"] == 2):
            self.fitness.predict_large()

    def initialize_model(self):
        model_path = self.para.nepin["model_init_path"]
        if model_path != None and os.path.exists(model_path):
            CurrentModel_state_dict = self.fitness.model.state_dict()
            InitModel_state_dict = torch.load(model_path,map_location=self.para.device)

            cut_stat_dict = dict( [(k, w) for k, w in InitModel_state_dict.items() if k in CurrentModel_state_dict])

            CurrentModel_state_dict.update(cut_stat_dict)
            self.fitness.model.load_state_dict(CurrentModel_state_dict, strict=False)


    def output_model(self):
        model_path = self.para.nepin["model_save_path"]
        torch.save(self.fitness.model.state_dict(), model_path)

    def output_check(self, steps):

        model_path = self.para.nepin["model_save_path"]
        dir_name, file_name = os.path.split(model_path)
        new_file_name = f"check_{steps}_{file_name}"
        new_model_path = os.path.join(dir_name, new_file_name)
        torch.save(self.fitness.model.state_dict(), new_model_path)
