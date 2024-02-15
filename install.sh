#!/usr/bin/env bash

###
 # @Author:   "  "
 # @Date: 2022-05-12 21:51:13
 # @LastEditTime: 2023-12-22 20:36:43
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /Cerberus-main/install.sh
 # email:  
### 

source /usr/local/miniconda3/etc/profile.d/conda.sh 


conda create -n ect python=3.7 -y

conda activate ect

echo "start install cudnn and pytroch ====================================";

conda install cudnn=8.0.5 -y
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y


echo "start install other package ===== ====================================";

pip install opencv-python
pip install timm==0.4.5

pip install tensorboardX
pip install wandb 


pip install IPython
pip install matplotlib

pip install tqdm
pip install scipy
pip install loguru
pip install h5py
echo " install  over ======================= ====================================";

python -c "import torch; print(torch.cuda.is_available(), 'cuda version : ',torch.version.cuda);"



pip install piq
