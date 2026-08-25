# WNM-3D Installation Guide

This guide describes the reference installation for **WNM-3D**, including the
model runtime, CUDA-accelerated dependencies, and the GN-Bench evaluation
stack.

> [!IMPORTANT]
> WNM-3D and GN-Bench use different PyTorch and CUDA versions. Install them in
> separate Conda environments as documented below.

## Environment Overview

| Component | Conda environment | Python | PyTorch | CUDA toolkit |
| --- | --- | ---: | ---: | ---: |
| WNM-3D policy server and training | `wnm_3d` | 3.11 | 2.8.0 | 12.9 |
| GN0 / GN-Bench evaluation | `bae` | 3.10 | 2.4.0 | 12.4 |

## Prerequisites

Before proceeding, ensure that the host system provides:

- A Linux environment with an NVIDIA GPU and a compatible NVIDIA driver
- [Git](https://git-scm.com/)
- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), Anaconda,
  or [Miniforge](https://github.com/conda-forge/miniforge)
- Sufficient local storage for model checkpoints, simulator assets, and the
  InteriorGS and GN-Matrix datasets

You can confirm that the NVIDIA driver is available with:

```bash
nvidia-smi
```

## Reference Hardware

The released Stage-III policy server has been validated on an NVIDIA H100
80 GB GPU. In the reference single-replica configuration
(`--num-replicas 1`), the loaded server used approximately 27 GiB of GPU
memory. This is an observed deployment figure rather than a strict minimum;
memory use can vary with the CUDA stack, attention implementation, and
inference settings.

The default server command launches eight independent replicas and therefore
expects eight visible GPUs. For a single-GPU deployment, pass
`--cuda-devices 0 --num-replicas 1`. The training launchers also default to
eight GPUs and retain the batch sizes and distributed settings used by the
project; other configurations should be validated for both memory use and
effective global batch size.

Each released stage checkpoint occupies approximately 26 GiB. Reserve
additional space for datasets, simulator assets, training checkpoints, and
optimizer states.

## 1. Install WNM-3D

### 1.1 Clone the Repository

```bash
git clone https://github.com/TeleHuman/WNM-3D.git
cd WNM-3D
```

If you already have a local checkout, run the remaining commands from the
repository root.

### 1.2 Create the Runtime Environment

Create a dedicated Python 3.11 environment:

```bash
conda create -n wnm_3d python=3.11 -y
conda activate wnm_3d
```

Upgrade the Python packaging toolchain before installing compiled
dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 1.3 Install PyTorch and the CUDA Toolkit

Install the reference PyTorch build for CUDA 12.9:

```bash
python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

Install the matching CUDA toolkit used to compile CUDA extensions:

```bash
conda install \
  -c "nvidia/label/cuda-12.9.0" \
  -c nvidia \
  -c conda-forge \
  cuda-toolkit=12.9 \
  -y
```

### 1.4 Install the Project

Install WNM-3D and its Python dependencies in editable mode:

```bash
python -m pip install -e .
```

For development and testing, install the optional development dependencies:

```bash
python -m pip install -e '.[dev]'
```

### 1.5 Install FlashAttention

Install FlashAttention without build isolation so that it compiles against the
PyTorch and CUDA versions in the active environment:

```bash
python -m pip install flash-attn --no-build-isolation
```

If a source build is not supported on your platform, consult the official
[FlashAttention release page](https://github.com/Dao-AILab/flash-attention/releases)
for compatible installation artifacts.

### 1.6 Verify the Installation

Confirm that PyTorch detects CUDA and that the WNM-3D package is importable:

```bash
python - <<'PY'
import torch
import gammanav

print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"WNM-3D package: {gammanav.__path__[0]}")
PY
```

For a GPU-enabled deployment, `CUDA available` should report `True`.

## 2. Install GN0 and GN-Bench

WNM-3D provides the policy server. The companion
[GN0](https://github.com/TeleHuman/GN0) repository provides the GN-Bench
simulator and evaluation workers.

From the parent directory of `WNM-3D`, clone GN0 and initialize the benchmark
tooling:

```bash
cd ..
git clone https://github.com/TeleHuman/GN0.git
cd GN0
git submodule update --init GN-Bench-Tools
```

Create a separate Python 3.10 environment for the simulator stack:

```bash
conda create -n bae python=3.10 -y
conda activate bae
python -m pip install --upgrade pip setuptools wheel
```

Install the GN-Bench reference PyTorch build and CUDA 12.4 toolkit:

```bash
python -m pip install \
  torch==2.4.0 \
  torchvision==0.19.0 \
  torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

conda install \
  -c "nvidia/label/cuda-12.4.0" \
  -c nvidia \
  -c conda-forge \
  cuda-toolkit=12.4 \
  -y
```

Install GN-Bench, GN0, and the matching `gsplat` build:

```bash
python -m pip install -e ./GN-Bench-Tools
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install gsplat \
  --index-url https://docs.gsplat.studio/whl/pt24cu124
```

For simulator-specific requirements and troubleshooting, refer to the
[GN0 installation guide](https://github.com/TeleHuman/GN0/blob/master/docs/INSTALLATION.md).
