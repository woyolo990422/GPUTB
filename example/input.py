import torch
nepin = {
    "dataset": [
                r"C:\Users\ddw\Desktop\work\GPUTB\example\data", #改成你的目录
                 ],
    "orbit": {"Si": ["3s","3p", "d*"]}, #轨道数 仅可以设spd *号为高能态轨道 帮助收敛
    "energy_level": {"Si":{"3s":0, "3p":0, "d*":20}}, #onsite能量偏移 （eV）
    "cutoff": 5.0, #截断半径 （埃）
    "optimizer": "AdamW", #优化器 
    "lambda_2": 0.0001,
    "lr": 0.001, #学习率 
    "generation": 100000,
    "batch_size": 1, 
    "seed": 2024,
    "plot_gap":1, #k点额外间隔 改为2则稀疏一倍
    "prediction":0, #0训练 1预测
    "intdtype": torch.int64, 
    "dtype": torch.float32, 
    "complexdtype": torch.complex64,
    "device": "cuda", #设备
    "decayfactor": 0.99975,
    "band_weight":[ [
                    1,1, 1,1 ,1,1#, 1,1,
                    #1,1, 1,1 ,1,1, 1,1,
                    #1,1, 1,1 ,1,1, 1,1, 
                    #1,1, 1,1 ,1,1, 1,1,
                    ],
],#能带权重 注意和能带数目一样
    "band_max": [6], #能带最大数目 
    "band_min": [0], #能带最小数目 
    "dos_sigma":0.1, #暂时未用dos loss 可以忽略
    "e_range": [-100,100], #能带训练loss的能量范围截断 
    "lr_method":"exp",
    "step_interval":50,
    "unit": "Hartree", #不用管 默认的
    "model_save_path": r"C:\Users\ddw\Desktop\work\GPUTB\example\3.pth", #改成你的目录
    "model_init_path": r"C:\Users\ddw\Desktop\work\GPUTB\example\2.pth", #改成你的目录
    "error_figures_path": r"C:\Users\ddw\Desktop\work\GPUTB\example", #改成你的目录
    "test_ratio":0, #测试数目 拿训练集中百分之多少测试 单个结构设为0 
    "des_norm":False, #是否用描述符归一化
    "basis_size": 32, #基函数展开维度
    "des_hidden": 32, #描述符隐藏层维度
    "n_max": 16, #描述符维度

    "embedding_init_val_0":0.005,#嵌入网络的初始化值
    "embedding_init_val_1":0.005,#嵌入网络的初始化值

    "param_init_val_0":0.005,#SK参数网络的初始化值
    "param_init_val_1":0.005,#SK参数网络的初始化值
}