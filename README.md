# Domain-Guided Weight Modulation for Semi-Supervised Domain Generalization - WACV 2025

<!-- [![paper](https://img.shields.io/badge/arXiv-Paper-42FF33)](https://arxiv.org/abs/2403.02782) 
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://bimsarapathiraja.github.io/mccl-project-page/)   -->

This repository gives the official implementation of [Domain-Guided Weight Modulation for Semi-Supervised Domain Generalization]() (WACV 2025)

## How to setup the environment

This code is built on top of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) and [ssdg-benchmark](https://github.com/KaiyangZhou/ssdg-benchmark). Please follow the instructions provided in https://github.com/KaiyangZhou/Dassl.pytorch and https://github.com/KaiyangZhou/ssdg-benchmark to install the `dassl` environment, as well as to prepare the datasets. 

## How to run

The script is provided in `/scripts/DGWM/run_ssdg.sh`. You need to update the `DATA` variable that points to the directory where you put the datasets. There are two input arguments: `DATASET` and `NLAB` (total number of labels).


Here we give an example. Say you want to run DGWM on OfficHome under the 10-labels-per-class setting (i.e. 1950 labels in total), run the following commands in your terminal,
```bash
conda activate dassl
cd scripts/DGWM
bash run_ssdg.sh ssdg_officehome 1950 
```

In this case, the code will run DGWM in four different setups (four target domains), each for five times (five random seeds). You can modify the code to run a single experiment instead of all at once if you have multiple GPUs.


To show the results, simply do
```bash
python parse_test_res.py output/ssdg_officehome/nlab_1950/DGWM/resnet18 --multi-exp
```

Check out our previous work on SSDG at [Towards Generalizing to Unseen Domains with Few Labels](https://arxiv.org/abs/2403.11674) (CVPR 2024).
