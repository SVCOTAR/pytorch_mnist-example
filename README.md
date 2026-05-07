# pytorch_mnist-example

本文档为基于pytorch的mnist的example，开启了gpu 加速。

# 1. 创建一个干净环境
conda create -n pytorchenv python=3.10 -y
# 2. 进入这个环境
conda activate pytorchenv
# 3.安装mamba
conda install mamba -c conda-forge

# 4. 在这个环境里安装 PyTorch + CUDA
mamba install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch

#5.如果能够上外网，直接使用pip安装：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
可以根据自身电脑的配置，从https://pytorch.org/官网下载。

运行python mnist.py
log如下：
<img width="362" height="119" alt="image" src="https://github.com/user-attachments/assets/adbad368-1992-492a-bd46-786ffd8d51f3" />

结果如下：
<img width="1086" height="588" alt="image" src="https://github.com/user-attachments/assets/fe92c2b9-2e82-4dac-a8c2-75b05d5cf0ba" />

