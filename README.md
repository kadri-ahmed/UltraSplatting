# 3D Gaussian Splatting for Ultrasound based Image datasets

This repository showcases the work done on 3D Gaussian Splatting for US images "Ultra-Splatting".

## Overview

We have the following folders: `datasets`, `gaussian-splatting`, `scripts`.

### datasets

Here you can find the code that transforms the tracked US datasets to colmap style datastructures. Additionally this is where we usually put the tracked datasets for the spine and liver.

### gaussian-splatting

The actual gaussian-splatting folder with some modifications to make our version run. Modifications are listed in the Modifications section at the end.

### scripts

A collection of some of the more important random scripts that we created during the semester. Specifically in the blender folder you can find all of the things to import bmode images in blender as well as the scenes to showcase these ideas. In the gaussian_sampler folder you can find an example for the torch based gaussian sampling based renderer we build and tested

## Running 3D Gaussian Splatting

0. First install and activate the necessary conda environment provided in gaussian_splatting
1. We start by generating a colmap dataset out of your tracked US data. For this we require both the images.npy and the poses.npy. Run `datasets/generate_colmap.py` and change the relative folder path in line 236 `relative_folder_path = "unprocessed/spine_phantom/left3_2/"` to the dataset you want to use. By default the data is output into `relative_folder_path/sparse/0` and `relative_folder_path/images`
2. Now you can run gaussian splatting on the dataset `python train.py -s /home/FOLDER_PATH/unprocessed/spine_phantom/left3_2`. If you want to train with a white background additionally provide the argument `--white_background` 

## Installation

### Requirements

Easiest setup is via WSL2 and creating a new Ubuntu20.04 instance. You also need Cuda, nvcc, miniconda and a c++ compiler.

```
wsl --list --online
wsl --install --distribution Ubuntu-20.04
```

If you are not using WSL you can also make it work on more modern versions of Cuda and Ubuntu as shown in this [github-issue](https://github.com/graphdeco-inria/gaussian-splatting/issues/923).

#### C++ Compiler

```
sudo apt update
sudo apt install build-essential gcc g++
```

#### Cuda

Installing [cuda 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive) via the instructions from the download archive.


Add cuda/nvcc to your paths:

```
echo 'export PATH=/usr/local/cuda-11.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Check if the installation was succesful by confirming that both your graphicsscard and your compiler are available:

```
nvcc --version
nvidia-smi
```


#### Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Cloning the repository and setting up the environment

Submodules as well as some example datasets should be contained within this repo so you can just clone it via:

```
git clone ...
```

Set up the environment using the provided environment.yml and conda after installing the requirements above:

```
cd gaussian_splatting
conda env create --file environment.yml
conda activate ultra_splatting
```

If it doesnt work it might be a gcc/g++ issue, try fixing it by manually setting a different gcc/g++ for installation as follows:

```
conda env remove -n ultra_splatting

sudo apt install gcc-9 g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDA_ROOT=/usr/local/cuda
ln -s /usr/bin/gcc-9 $CUDA_ROOT/bin/gcc
ln -s /usr/bin/g++-9 $CUDA_ROOT/bin/g++
```

## Modifications

The Projection steps were modified following this [github-issue](https://github.com/graphdeco-inria/gaussian-splatting/issues/578). This means `gaussian-splatting/utils/graphics_utils` `getProjectionMatrix` function was modified to use an orthographic matrix based on the fov provided by our colmap dataset generator. Additionally we also manually need to change the calculation of the Jacobian Matrix in `gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` via the function `computeCov2D`
