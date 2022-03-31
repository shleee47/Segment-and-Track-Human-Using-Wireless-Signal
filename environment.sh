conda create -y -n wifi python=3.8
conda activate wifi
conda install scipy
​
####select according to your conda version####
####https://pytorch.org/####
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
​
pip install PyYAML
conda install -c conda-forge tqdm
pip install tensorboard
