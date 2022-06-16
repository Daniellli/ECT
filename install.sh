#!/usr/bin/env bash

###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:51:13
<<<<<<< HEAD
 # @LastEditTime: 2022-06-14 21:19:08
=======
 # @LastEditTime: 2022-06-14 21:58:57
>>>>>>> cb0a9a0a22cf34a879158bc461854d18fe6035a3
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/install.sh
 # email: xushaocong@stu.xmu.edu.cn
### 

# source /usr/local/miniconda3/bin/activate



#todo :
# conda update -n base -c defaults conda

# python -V 

# conda create -n cerberus3 python=3.7 -y
# source activate
# conda deactivate
# conda activate cerberus3

echo "start install cudnn and pytroch ====================================";

# conda install cudnn=8.0.4 -y
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# #* install cudnn
# conda install cudnn=8.2.1 -y
# #* install pytroch and cuda 11.3 
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y


conda install cudnn=8.0.5 -y
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y


echo "start install other package ===== ====================================";


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
echo " install  over ======================= ====================================";
python -c "import torch; print(torch.cuda.is_available(), 'cuda version : ',torch.version.cuda);"


# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# conda config --set show_channel_urls yes
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# pip install pip -U
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import sys; print(sys.path)"