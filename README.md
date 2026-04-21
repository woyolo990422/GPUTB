# GPUTB

[中文说明](README.zh-CN.md)

GPUTB is an open-source package for building **orthogonal tight-binding Hamiltonians** from **machine-learned Slater-Koster (SK) parameters**. The current codebase supports band-structure fitting against reference electronic-structure data and can also export sparse real-space Hamiltonian data for LSQT-style transport workflows.

The project is developed by Prof. Jian Sun's group at the School of Physics, Nanjing University: <https://sun.nju.edu.cn>

## What the current code provides

- Train a neural tight-binding model from reference band structures.
- Predict band structures from a saved model checkpoint.
- Export sparse Hamiltonian tensors for large-scale LSQT post-processing.
- Support multi-orbital basis definitions such as `s`, `p`, `d`, as well as extra empty orbitals labeled like `d*`.

## Repository layout

```text
GPUTB-main/
|- TB/            # training, prediction, band fitting
|- LSQT/          # sparse Hamiltonian export and LSQT preparation
|- utilities/     # shared utilities, orbital index mapping, onsite database
|- example/       # sample input.py and example dataset
`- README.md
```

## Dependencies

Install the Python packages used directly by the code:

```bash
pip install "torch<=2.6.0" numpy ase numba h5py matplotlib torch_geometric
```

Notes:

- `torch_geometric` must match your PyTorch and CUDA build.
- If you do not have a CUDA device, set `device` to `"cpu"` in your input file.
- The code imports `matplotlib` during prediction, so install it even if you only plan to evaluate a model.

## Training data layout

Each training dataset directory is expected to contain:

- `xdat.traj`: an ASE trajectory with one or more structures.
- `kpoints.npy`: a NumPy array of shape `(nk, 3)`.
- `eigs.npy`: a NumPy array of shape `(nframe, nk, nband)` or `(nk, nband)`.

The bundled example under `example/data/` follows this layout. In the current sample:

- `kpoints.npy` has shape `(302, 3)`.
- `eigs.npy` has shape `(1, 302, 14)`.

## Quick start: training or prediction

The training entry point is `TB/main.py`. It loads a dictionary named `nepin` from an `input.py` file in the **current working directory**.

1. Edit `example/input.py`.
2. Replace the absolute paths in that file with paths on your machine.
3. Set `device="cpu"` if CUDA is unavailable.
4. From the `example/` directory, run:

```bash
python ../TB/main.py
```

### Important `nepin` fields

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

Key behaviors inferred from the code:

- `prediction = 0`: training mode.
- `prediction = 1`: prediction mode, compares predicted and reference bands.
- `prediction = 2`: large-system export path for sparse k-space Hamiltonian data.
- `model_save_path`: checkpoint saved every `step_interval` iterations.
- `model_init_path`: optional checkpoint loaded before training or prediction.
- `band_min`, `band_max`, and `band_weight`: define the fitted band window and loss weights.
- `test_ratio`: fraction of structures used as test data.

## Outputs

During training:

- Progress is printed to stdout every `step_interval`.
- The main checkpoint is written to `model_save_path`.
- Extra snapshots named like `check_1000_<model_name>` are written every 1000 steps.

During prediction:

- Error figures are saved into `error_figures_path`.
- Scatter and line data are also exported as text files for post-processing.

## LSQT workflow

The LSQT entry point is `LSQT/main.py`. It loads a dictionary named `runin` from `lsqt_input.py` in the current working directory.

Typical workflow:

1. Prepare a trained model checkpoint.
2. Create a working directory containing `lsqt_input.py`.
3. Run:

```bash
python ../LSQT/main.py
```

The current LSQT code expects fields such as:

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

The exported files are assembled into `model.h5` under `dump_sparseH`, with intermediate `hoppings.h5` and `onsites.h5` generated and then merged by `LSQT/Run.py`.

## Practical notes

- Orbital labels are parsed by their letter part for angular momentum, while labels like `d*` are used as extra empty orbitals and remain valid keys for onsite energies.
- `example/input.py` is a template, not a ready-to-run config; its paths point to the original developer environment.
- The code is script-oriented rather than packaged, so running from the correct working directory matters.

## Citation

If you use GPUTB in research, please cite:

```text
Wang, Y. et al. GPUTB: Efficient machine learning tight-binding method for
large-scale electronic properties calculations. Computational Materials Today,
8, 100039 (2025). https://doi.org/10.1016/j.commt.2025.100039
```
