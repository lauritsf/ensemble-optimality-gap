# On Joint Regularization and Calibration in Deep Ensembles

This repository contains the code for the TMLR 2025 paper:
**On Joint Regularization and Calibration in Deep Ensembles**
*Laurits Fredsgaard, Mikkel N. Schmidt*

[**Paper (TMLR/OpenReview)**](https://openreview.net/forum?id=6xqV7DP3Ep) | [**Paper (PDF)**](https://openreview.net/pdf?id=6xqV7DP3Ep)

## Requirements

The code can be run on a local machine (with CUDA or CPU) or on the LUMI supercomputer.

### General Setup (CUDA / CPU)

1. **Install PyTorch:** Follow the official instructions to install `torch` and `torchvision` for your specific platform (CUDA, CPU):
    [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

2. **Install PyTorch Geometric:** Follow the official instructions, making sure to select the version compatible with your PyTorch install:
    [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

3. **Install remaining packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install the project:**

    ```bash
    # Use -e for editable mode, or omit it for a standard install
    pip install -e .
    ```

### LUMI Setup (ROCm)

The experiments for the paper were run on LUMI using the environment and steps below.

**Container:** `lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1-dockerhash-04f2083a6cb0.sif`

**Setup Steps:**

```bash
# 0. Load Singularity module
# E.g. https://github.com/lauritsf/cli-tools/blob/8cef9fd62eecf22d5c8193532505accd46b72c58/modules/singularity-bindings-lumi/2024-11-28.lua
module load singularity-bindings-lumi

# 1. Set path to the container and start an interactive shell
SIF=/path/to/lumi-pytorch-rocm-6.1.3-python-3.12-pytorch-v2.4.1-dockerhash-04f2083a6cb0.sif
singularity shell $SIF

# 2. Activate the included Conda environment
$WITH_CONDA

# 3. Create a virtual environment inside the container
# (The --system-site-packages flag is essential to access the container's PyTorch)
python -m venv .venv --system-site-packages

# 4. Activate the virtual environment
source .venv/bin/activate

# 4. Install the exact (LUMI-specific) packages
python -m pip install -r requirements-lumi.txt

# 5. Install the project in editable mode
pip install -e .
```

## How to Cite

If you use this code or our findings in your research, please cite the paper:

```bibtex
@article{
fredsgaard2025on,
title={On Joint Regularization and Calibration in Deep Ensembles},
author={Laurits Fredsgaard and Mikkel N. Schmidt},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={[https://openreview.net/forum?id=6xqV7DP3Ep](https://openreview.net/forum?id=6xqV7DP3Ep)},
note={}
}
```
