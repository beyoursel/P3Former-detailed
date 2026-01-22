#!/bin/bash

# first install pytorch 1.10-cu11
conda create -n p3former python==3.8 -y
conda activate p3former
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install openmim
mim install mmengine==0.7.3
mim install mmcv==2.0.0rc4
mim install mmdet==3.0.0
mim install mmdet3d==1.1.0
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install "yapf==0.32.0"
pip install setuptools==59.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# create env
conda create -n p3former python=3.10 -y
# install torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
# install mmdet3d
pip install --no-deps mmengine==0.7.3 mmdet==3.0.0 mmsegmentation==1.0.0
pip install --no-deps --no-build-isolation git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html --no-deps
 
wget https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.0.9-cp310-cp310-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp310-cp310-linux_x86_64.whl 
