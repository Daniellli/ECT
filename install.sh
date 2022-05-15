###
 # @Author: xushaocong
 # @Date: 2022-05-12 21:51:13
 # @LastEditTime: 2022-05-12 22:40:43
 # @LastEditors: xushaocong
 # @Description: 
 # @FilePath: /Cerberus-main/install.sh
 # email: xushaocong@stu.xmu.edu.cn
### 


# conda create -n cerberus2 python=3.7 -y
# conda activate cerberus2
# #* 所以不一定要1.8.1 , 1.8.0,torchvision==0.9.0也可以
# # pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge


python -c "import torch; print(torch.cuda.is_available())"
pip install opencv-python
pip install timm==0.4.5
pip install tensorboardX
pip install wandb 
pip install IPython
pip install matplotlib










# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
# conda config --set show_channel_urls yes

# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
