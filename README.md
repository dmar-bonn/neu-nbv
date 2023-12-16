# NeU-NBV: Next Best View Planning Using Uncertainty Estimation in Image-Based Neural Rendering

Liren Jin, Xieyuanli Chen, Julius Rückin, Marija Popovic<br>
University of Bonn

This repository contains the implementation of our paper "NeU-NBV: Next Best View Planning Using Uncertainty Estimation in Image-Based Neural Rendering" accepted to IROS 2023.


```commandline
@INPROCEEDINGS{jin2023neunbv,
      title={NeU-NBV: Next Best View Planning Using Uncertainty Estimation in Image-Based Neural Rendering}, 
      booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
      author={Jin, Liren and Chen, Xieyuanli and Rückin, Julius and Popović, Marija},
      pages={11305-11312},
      year={2023},
      doi={10.1109/IROS55552.2023.10342226}}
```

## Abstract

Autonomous robotic tasks require actively perceiving the environment to achieve application-specific goals. In this paper, we address the problem of positioning an RGB camera to collect the most informative images to represent an unknown scene, given a limited measurement budget. We propose a novel mapless planning framework to iteratively plan the next best camera view based on collected image measurements. A key aspect of our approach is a new technique for uncertainty estimation in image-based neural rendering, which guides measurement acquisition at the most uncertain view among view candidates, thus maximising the information value during data collection. By incrementally adding new measurements into our image collection, our approach efficiently explores an unknown scene in a mapless manner. We show that our uncertainty estimation is generalisable and valuable for view planning in unknown scenes. Our planning experiments using synthetic and real-world data verify that our uncertainty-guided approach finds informative images leading to more accurate scene representations when compared against baselines.

An overview of our NBV planning framework:
![Framework](media/images/framework.png)

## Dataset
- Download [DTU dataset](https://phenoroam.phenorob.de/file-uploader/download/public/953455041-dtu_dataset.zip) and [Shapenet](https://phenoroam.phenorob.de/file-uploader/download/public/731944960-shapenet.zip) dataset to scripts/neural_rendering/data/dataset folder.

## Environment Setup
Clone the repo to your local. We use docker to make deployment in your machine easier (hopefully). Note that for training and planning experiments on DTU dataset, docker is not necessary, you can also just use conda environment.

1. Build docker image
    ```commandline
    cd neu-nbv
    docker build . -t neu-nbv:v1
    ```
2. To use gpu in docker container (docker-compose-gpu.yaml), follow [nvidia-run-time support](https://nvidia.github.io/nvidia-container-runtime/) to set up runtime support.

3. For network training, you can either activate conda environment or start a docker container.
 - In conda environment
    ```commandline
    conda env create -f environment.yaml
    conda activate neu-nbv
    ```
 - In docker container
    ```commandline
    make up
    make training
    ```
Then follow Network Training section.

## Network Training
For starting training process,
```commandline
cd scripts/neural_rendering
```
```commandline
python train.py -M <model name> --setup_cfg_path <path to training setup file>
```
Continue training:
```commandline
python train.py -M <model name> --setup_cfg_path <path to training setup file> --resume
```
Visualize training progress via:
``` commandline
tensorboard --logdir <project dir>/logs/<model name>
```

We also provide two pretrained models trained on [DTU](https://phenoroam.phenorob.de/file-uploader/download/public/195880506-dtu_training.zip) and [Shapenet](https://phenoroam.phenorob.de/file-uploader/download/public/196062945-shapenet_training.zip). Copy these folder to scripts/neural_rendering/logs.

## Planning Experiments
### On DTU dataset
Either use conda environment or docker container. 
```commandline
cd scripts
python planning/dtu_experiment.py -M <model name>
```
### In Simulator
We first need to set MODEL_NAME to choose which model we spawn in our simulator, currently only "car" and "indoor" are supported. 
```commandline
make up
export MODEL_NAME=<model name>
make simulator
```
Before starting experiments, we run random planner to get 200 test views. To generate test data, in a new terminal,
```commandline
make planning 
cd scripts
python planning/simulator_planning -P random -BG 200
```
We can start planning experiments in simulator, 
```commandline
python planning/simulator_experiment -M <model name> -test_data_path <path to test data>
```
Shut dowm containers,
```commandline
make down
```

## Plot Results
```commandline
cd scripts
python utils/plot.py --dataframe_path <path to the dataframe>
```

## Acknowledgements
Parts of the code were based on [pixelNeRF](https://github.com/sxyu/pixel-nerf.git) and [NeuralMVS](https://github.com/AIS-Bonn/neural_mvs.git).

## Maintainer
Liren Jin, ljin@uni-bonn.de


## Project Funding
This work has been fully funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy, EXC-2070 – 390732324 (PhenoRob) and supported by the NVIDIA Academic Hardware Grant Program. All authors are with the Institute of Geodesy and Geoinformation, University of Bonn.

