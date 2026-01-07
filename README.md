# MIP: Dual-Temporal Mamba Inertial Poser

Official repository for:
Mamba Inertial Poser: Dual-Temporal Causal Human Motion Tracking from Sparse Inertial Sensors

This repository provides the official implementation of MIP, a dual-temporal
state-space model for real-time full-body human motion capture from sparse IMU
sensors.

The codebase will be progressively released to support full reproducibility of
all experiments reported in the paper.

########
# Setup

We use **Python 3.8.10**.  
Please install the following dependencies:

- chumpy  
- open3d  
- pybullet  
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

- Student model: download from [here](https://github.com/HanShijia999/MIP/releases/download/V1.0.0/best_distill_select2.pt)  
  Place to:
  ```bash
  NetBank/weights/best_distill_select2.pt
  ```

- Pretrained PNP weights: download from [here](https://github.com/HanShijia999/MIP/releases/download/V1.0.0/weights.pt)  
  Place to:
  ```bash
  data/weights/PNP/weights.pt
  ```

---

## Prepare Test Datasets

Download the preprocessed DIP-IMU and TotalCapture test datasets from [here](https://github.com/HanShijia999/MIP/releases/download/V1.0.0/dipimu.pt), and place to:

```bash
data/test_datasets/dipimu.pt
```

> By downloading these datasets, you agree to the original DIP-IMU license terms:  
> https://dip.is.tue.mpg.de
#######
Setup
-----
We use python 3.8.10. You should install chumpy open3d pybullet qpsolvers numpy-quaternion vctoolkit==0.1.5.39 and pytorch with CUDA (we use pytorch 2.0.1 with CUDA 11.8). You also need to compile and install rbdl 2.6.0 with python bindings and urdf reader addon.

If chumpy reports errors, comment the lines from numpy import bool ... that generate errors.

If quadprog solver is not found, install with pip install qpsolvers[quadprog]

Then:
cd kernels/selective_scan && pip install -e .

Prepare SMPL body model

    Download SMPL model from here. You should click SMPL for Python and download the version 1.0.0 for Python 2.7 (10 shape PCs). Then unzip it. (link:smpl.is.tue.mpg.de)
    Rename and put the male model file into models/SMPL_male.pkl.

Prepare physics body model

    Download the physics body model from here and unzip it. (link:https://github.com/HanShijia999/MIP/releases/download/V1.0.0/models.zip)
    Rename and put the files into models/physics.urdf, models/plane.obj, models/plane.urdf.

The physics body model and ground plane are adapted from PhysCap and PNP.
Original sources: https://github.com/Xinyu-Yi/PNP?tab=readme-ov-file#prepare-smpl-body-model

Download pre-trained weights
    https://github.com/HanShijia999/MIP/releases/download/V1.0.0/best_distill_select2.pt
    put to NetBank/weights/best_distill_select2.pt
Download pre-trained PNP weights
    https://github.com/HanShijia999/MIP/releases/download/V1.0.0/weights.pt
    put to data/weights/PNP/weights.pt

Prepare test datasets
    Download the preprocessed DIP-IMU and TotalCapture dataset (with two different calibrations as listed in the paper) from here. Please note that by downloading the preprocessed datasets you agree to the same license conditions as for the DIP-IMU dataset (https://dip.is.tue.mpg.de/) (link:https://github.com/HanShijia999/MIP/releases/download/V1.0.0/dipimu.pt)
    ename and put the files into data/test_datasets/dipimu.pt




Planned Release
---------------

[◌] Dual-Temporal Mamba backbone and PQN modules  
[◌] Pretrained models and demo scripts  
[ ] Training and distillation code  
[ ] Evaluation and visualization tools  


Contact
-------

Corresponding Author:
Jianfei Yang
jianfei.yang@ntu.edu.sg
