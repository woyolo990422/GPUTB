import torch
import os
from Parameters import Parameters
from Fitness import Fitness
from utilities.Index import Index
import time
from typing import Union

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

    def convert_model_dtype(self,
        model: torch.nn.Module,
        param_dtype: Union[torch.dtype, None] = torch.float64,
        buffer_dtype: Union[torch.dtype, None] = torch.float64,
        complex_to: Union[torch.dtype, None] = torch.complex128
    ) -> torch.nn.Module:

        def _convert(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
            if tensor.is_complex() and complex_to is not None:
                return tensor.to(complex_to)
            elif not tensor.is_complex() and target_dtype is not None:
                return tensor.to(target_dtype)
            return tensor

        for name, param in model.named_parameters(recurse=False):
            if param is not None:
                model._parameters[name] = _convert(param, param_dtype)

        for name, buf in model.named_buffers(recurse=False):
            if buf is not None:
                model._buffers[name] = _convert(buf, buffer_dtype)

        for submodule in model.children():
            self.convert_model_dtype(submodule, param_dtype, buffer_dtype, complex_to)

        return model


    def initialize_model(self):
        model_path = self.para.model_init_path
        if model_path is None or not os.path.exists(model_path):
            return

        saved_data = torch.load(model_path, map_location=self.para.device)

        # 关键检测逻辑：判断是旧版还是新版格式
        if isinstance(saved_data, dict) and 'model_state' in saved_data:  # 新版格式
            model_state = saved_data['model_state']
            gradients = saved_data.get('gradients', {})
            optimizer_state = saved_data.get('optimizer_state', None)
        else:  # 旧版直接保存state_dict的格式
            model_state = saved_data  # 旧版数据直接作为模型参数
            gradients = {}  # 旧版无梯度
            optimizer_state = None  # 旧版无优化器状态

        # 加载模型参数
        current_model_state = self.fitness.model.state_dict()
        filtered_model_state = {k: v for k, v in model_state.items() if k in current_model_state}
        current_model_state.update(filtered_model_state)
        self.fitness.model.load_state_dict(current_model_state, strict=False)
        self.fitness.model = self.convert_model_dtype(self.fitness.model,
        param_dtype=torch.float32,    # 将 float64 参数转为 float64
        buffer_dtype=torch.float32,   # 将 float64 缓冲区转为 float64
        complex_to=torch.complex64   # 将 complex64 转为 complex128
            )


        # 加载优化器状态（旧版文件无优化器则跳过）
        if optimizer_state is not None and self.fitness.optimizer is not None:
            current_lrs = [group['lr'] for group in self.fitness.optimizer.param_groups]
            self.fitness.optimizer.load_state_dict(optimizer_state)
            for i, lr in enumerate(current_lrs):
                self.fitness.optimizer.param_groups[i]['lr'] = lr  # 保持当前学习率

    def output_model(self):
        model_path = self.para.model_save_path
        # 保存为新版格式（包含梯度+优化器）
        torch.save({
            'model_state': self.fitness.model.state_dict(),
            'gradients': {n: p.grad.clone() if p.grad is not None else None
                        for n, p in self.fitness.model.named_parameters()},
            'optimizer_state': self.fitness.optimizer.state_dict() if self.fitness.optimizer else None
        }, model_path)

    def output_check(self, steps):

        model_path = self.para.nepin["model_save_path"]
        dir_name, file_name = os.path.split(model_path)
        new_file_name = f"check_{steps}_{file_name}"
        new_model_path = os.path.join(dir_name, new_file_name)
        torch.save(self.fitness.model.state_dict(), new_model_path)
