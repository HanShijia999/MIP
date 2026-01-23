# MIP: Dual-Temporal Mamba Inertial Poser

Official repository for:
Mamba Inertial Poser: Dual-Temporal Causal Human Motion Tracking from Sparse Inertial Sensors

This repository provides the official implementation of MIP, a dual-temporal
state-space model for real-time full-body human motion capture from sparse IMU
sensors.

The codebase will be progressively released to support full reproducibility of
all experiments reported in the paper.

# Setup

We use **Python 3.8.10**.  
Please install the following dependencies:

- chumpy  
- open3d  
- pybullet  
- einops
- timm
- fvcore
- qpsolvers  
- numpy-quaternion  
- vctoolkit==0.1.5.39  
- PyTorch (2.0.1 + CUDA 11.8)

You also need to compile and install **RBDL 2.6.0** with its Python bindings and the URDF reader addon.

> If `chumpy` reports errors, comment out the lines `from numpy import bool, ...` that cause the exception.  
> If the `quadprog` solver is missing, install it via:
> ```bash
> pip install qpsolvers[quadprog]
> ```

Then install the selective scan kernel:

```bash
cd kernels/selective_scan
pip install -e .
```
in win:
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.29
set DISTUTILS_USE_SDK=1
set MSSdk=1
conda activate py38

cd /d D:\work\MIP\kernels\selective_scan
rmdir /s /q build
rmdir /s /q selective_scan.egg-info

pip install -e . > build_v142_real.log 2>&1
---

## Prepare SMPL Body Model

Download the SMPL model from the official website [here](https://smpl.is.tue.mpg.de).  
Please download **SMPL for Python (v1.0.0, 10 shape PCs)**, unzip the package, and place the male model file as:

```bash
models/SMPL_male.pkl
```

---

## Prepare Physics Body Model

We provide the physics body model used in MIP as an official release asset.

Download the physics model package from [here](https://github.com/HanShijia999/MIP/releases/download/V1.0.0/models.zip), unzip it, and place the files as:

```bash
models/physics.urdf
models/plane.obj
models/plane.urdf
```

> The physics body model and the ground plane are adapted from PhysCap and PNP.  
> Original source: https://github.com/Xinyu-Yi/PNP

---

## Download Pretrained Models

- Student model: download from [here](https://github.com/HanShijia999/MIP/releases/download/V1.0.0/MIP_weights.pt)  
  Place to:
  ```bash
  data/weights/MIP_weights.pt
  ```

---

## Prepare Test Datasets

Download the preprocessed DIP-IMU and TotalCapture test datasets from [here](https://github.com/HanShijia999/MIP/releases/download/V1.0.0/dipimu.pt), and place to:

```bash
data/test_datasets/dipimu.pt
```

> By downloading these datasets, you agree to the original DIP-IMU license terms:  
> https://dip.is.tue.mpg.de



# Planned Release
---------------

- [✅] Basic runable code
- [✅] Pretrained models and demo scripts
- [ ] Extra visualization tools  


Contact
-------

Corresponding Author:
Jianfei Yang
jianfei.yang@ntu.edu.sg
