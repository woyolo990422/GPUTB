# GPUTB

[English](README.md)

GPUTB 是一个基于**机器学习 Slater-Koster (SK) 参数**构建**正交紧束缚哈密顿量**的开源代码。按照当前仓库里的实现，它主要支持两类任务：

- 使用参考能带数据训练神经网络紧束缚模型。
- 导出面向 LSQT 大规模输运计算的稀疏实空间哈密顿量数据。

项目来自南京大学物理学院孙建课题组：<https://sun.nju.edu.cn>

## 当前代码支持的能力

- 以参考能带为监督信号训练 TB 模型。
- 载入已有模型进行能带预测与误差分析。
- 生成 LSQT 后处理中使用的稀疏哈密顿量文件。
- 支持 `s`、`p`、`d` 等多轨道基组，也支持 `d*` 这类额外空轨道标签。

## 仓库结构

```text
GPUTB-main/
|- TB/            # 训练、预测、能带拟合
|- LSQT/          # 稀疏哈密顿量导出与 LSQT 预处理
|- utilities/     # 通用工具、轨道索引、onsite 数据库
|- example/       # 示例 input.py 和示例数据
`- README.md
```

## 依赖环境

根据当前代码直接导入的库，可以先安装：

```bash
pip install "torch<=2.6.0" numpy ase numba h5py matplotlib torch_geometric
```

说明：

- `torch_geometric` 需要和你的 PyTorch / CUDA 版本匹配。
- 如果本机没有 CUDA，请在输入文件里把 `device` 改成 `"cpu"`。
- 预测流程会直接导入 `matplotlib`，所以只做推理也建议安装它。

## 训练数据目录格式

每个数据集目录需要包含：

- `xdat.traj`：ASE 轨迹文件，支持一个或多个结构帧。
- `kpoints.npy`：形状为 `(nk, 3)` 的 NumPy 数组。
- `eigs.npy`：形状为 `(nframe, nk, nband)` 或 `(nk, nband)` 的 NumPy 数组。

仓库自带的 `example/data/` 就是一个最小示例。当前示例中：

- `kpoints.npy` 的形状是 `(302, 3)`。
- `eigs.npy` 的形状是 `(1, 302, 14)`。

## 快速开始：训练或预测

训练入口是 `TB/main.py`。这个脚本会从**当前工作目录**读取 `input.py`，并从里面加载名为 `nepin` 的字典。

推荐步骤：

1. 修改 `example/input.py`。
2. 把其中写死的绝对路径改成你机器上的实际路径。
3. 如果没有 GPU，把 `device="cuda"` 改成 `device="cpu"`。
4. 在 `example/` 目录下执行：

```bash
python ../TB/main.py
```

### `nepin` 里最重要的字段

```python
nepin = {
    "dataset": [r"/path/to/dataset_dir"],
    "orbit": {"Si": ["3s", "3p", "d*"]},
    "energy_level": {"Si": {"3s": 0, "3p": 0, "d*": 20}},
    "cutoff": 5.0,
    "generation": 100000,
    "prediction": 0,
    "device": "cuda",
    "model_save_path": r"/path/to/model.pth",
    "model_init_path": None,
    "error_figures_path": r"/path/to/output_dir",
}
```

结合代码可以确认：

- `prediction = 0`：训练模式。
- `prediction = 1`：预测模式，会和参考能带做对比。
- `prediction = 2`：大体系导出路径，用于生成稀疏 k 空间哈密顿量相关文件。
- `model_save_path`：每隔 `step_interval` 会保存一次模型。
- `model_init_path`：可选，用来载入已有权重继续训练或直接预测。
- `band_min`、`band_max`、`band_weight`：共同决定参与拟合的能带范围与损失权重。
- `test_ratio`：从结构集中划分测试集的比例。

## 输出结果

训练阶段：

- 每隔 `step_interval` 在终端打印一次训练/测试误差。
- 模型主检查点写入 `model_save_path`。
- 每 1000 步额外生成一次 `check_1000_xxx.pth` 这类快照。

预测阶段：

- 在 `error_figures_path` 下保存误差图。
- 同时导出散点图和折线图对应的文本数据，便于后处理。

## LSQT 使用方式

LSQT 入口是 `LSQT/main.py`。它会从当前工作目录读取 `lsqt_input.py`，并加载其中的 `runin` 字典。

典型流程：

1. 先准备好训练完成的模型参数文件。
2. 新建一个工作目录，并放入 `lsqt_input.py`。
3. 在该目录下执行：

```bash
python ../LSQT/main.py
```

按当前代码，`runin` 通常需要提供这些字段：

- `orbit`
- `energy_level`
- `cutoff`
- `device`
- `model_init_path`
- `structure_path`
- `dump_sparseH`
- `compute_sparseH`
- `transport_direction`
- `nn_split_scale`
- `lsqt_split_scale`
- `time_step`
- `num_of_steps`
- `num_moments`
- `num_energies`
- `start_energy`
- `end_energy`
- `max_energy`

导出过程中会先在 `dump_sparseH` 下生成中间文件 `hoppings.h5`、`onsites.h5`，然后由 `LSQT/Run.py` 合并整理到最终的 `model.h5`。

## 实用提醒

- 轨道角动量是按标签里的字母部分解析的，因此 `d*` 这类额外空轨道写法在代码里是允许的，并且可以作为 onsite 能级字典的键。
- `example/input.py` 更像模板，不是开箱即用配置；其中路径仍然指向原作者本地环境。
- 这套代码更偏脚本式使用，没有做成安装包，因此“从哪个目录运行”很关键。

## 引用

如果你在研究工作中使用 GPUTB，可以引用：

```text
Wang, Y. et al. GPUTB: Efficient machine learning tight-binding method for
large-scale electronic properties calculations. Computational Materials Today,
8, 100039 (2025). https://doi.org/10.1016/j.commt.2025.100039
```
