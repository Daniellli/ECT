#!/usr/bin/env bash

###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:51:13
 # @LastEditTime: 2022-06-13 15:49:53
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/install.sh
 # email: xushaocong@stu.xmu.edu.cn
### 

# source /usr/local/miniconda3/bin/activate

# python -V 
# conda create -n cerberus python=3.7 -y
source activate
conda deactivate
conda activate cerberus



conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python -c "import torch; print(torch.cuda.is_available())"
pip install opencv-python
pip install timm==0.4.5
#!+=============
pip install tensorboardX
pip install wandb 
#!+=============
pip install IPython
pip install matplotlib

pip install tqdm
pip install scipy
pip install loguru
pip install h5py


# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# conda config --set show_channel_urls yes
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# pip install pip -U
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple