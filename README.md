# MIP: Dual-Temporal Mamba Inertial Poser

Official repository for:
Mamba Inertial Poser: Dual-Temporal Causal Human Motion Tracking from Sparse Inertial Sensors

This repository provides the official implementation of MIP, a dual-temporal
state-space model for real-time full-body human motion capture from sparse IMU
sensors.

The codebase will be progressively released to support full reproducibility of
all experiments reported in the paper.

Install dependencies
-----
We use python 3.8.10. You should install chumpy open3d pybullet qpsolvers numpy-quaternion vctoolkit==0.1.5.39 and pytorch with CUDA (we use pytorch 2.0.1 with CUDA 11.8). You also need to compile and install rbdl 2.6.0 with python bindings and urdf reader addon.

If chumpy reports errors, comment the lines from numpy import bool ... that generate errors.

If quadprog solver is not found, install with pip install qpsolvers[quadprog]

Then:
cd kernels/selective_scan && pip install -e .

Prepare SMPL body model

    Download SMPL model from here. You should click SMPL for Python and download the version 1.0.0 for Python 2.7 (10 shape PCs). Then unzip it.
    Rename and put the male model file into models/SMPL_male.pkl.

Prepare physics body model

    Download the physics body model from here and unzip it.
    Rename and put the files into models/physics.urdf, models/plane.obj, models/plane.urdf.

The physics body model and ground plane are adapted from PhysCap and PNP.
Original sources: https://github.com/Xinyu-Yi/PNP?tab=readme-ov-file#prepare-smpl-body-model

copy models, NetBank/weights, data
make requirements.txt


Planned Release
---------------

[ ] Dual-Temporal Mamba backbone and PQN modules  
[ ] Pretrained models and demo scripts  
[ ] Training and distillation code  
[ ] Evaluation and visualization tools  


Contact
-------

Corresponding Author:
Jianfei Yang
jianfei.yang@ntu.edu.sg
