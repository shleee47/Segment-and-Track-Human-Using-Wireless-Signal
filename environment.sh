conda create -y -n gen python=3.7.7
conda activate gen
conda install scipy
​
####select according to your conda version####
####https://pytorch.org/####
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
​
conda install -c conda-forge librosa
pip install PyYAML
conda install -c conda-forge tqdm
pip install tensorboard
pip install torchlibrosa