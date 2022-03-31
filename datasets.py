import json
import sys
sys.path.append('..')
from torch.utils.data.dataset import Dataset
from pathlib import Path
import pickle
import pdb
import torch
import numpy as np
import argparse
import os
import sys
import librosa
import numpy as np
import scipy.io as sio
from scipy import signal
from tqdm import tqdm
from features import Feature_Extractor
from scipy import io as sio
import warnings
warnings.filterwarnings("ignore")

class Audio_Reader(Dataset):

    def __init__(self, datalist,labelist):
        super(Audio_Reader, self).__init__()
        self.datalist = datalist
        self.labelist = labelist
        self.augment = False

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        data_path = self.datalist[idx]
        label_path = self.labelist[idx]

        '''Load CSI data & label'''
        mat_file = data_path.split('/')[-1]+'_17.mat'
        data = sio.loadmat(os.path.join(data_path, mat_file))['csi']
        label = np.load(os.path.join(label_path,'label.npy'))

        return data.transpose(2,0,1), label


def Audio_Collate(batch):
   
    pdb.set_trace()
    data, label = list(zip(*batch))
    data_len = torch.LongTensor(np.array([x.size(1) for x in data if x.size(1)!=1]))

    max_len = max(data_len)
    wrong_indices = []
    
    B = len(data)

    inputs = torch.zeros(B, 150, max_len, 3, 3)
    labels = torch.zeros(B, max_len, 224, 224)
    j = 0

    '''zero pad'''    
    for i in range(B):
        TT= data[i].size(1)
        inputs[j, : , :TT,:] = data[i]
        labels[j, :TT,:] = class_num[i]
        j += 1

    data = (inputs, labels, data_len)
    return data


class Test_Reader(Dataset):

    def __init__(self, datalist, labelist):
        super(Test_Reader, self).__init__()
        self.datalist = datalist
        self.labelist = labelist
        self.augment = False

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):

        pdb.set_trace()
        data_path = self.datalist[idx]
        label_path = self.labelist[idx]

        '''Load CSI data & label'''
        mat_file = data_path.split('/')[-1]+'_17.mat'
        data = sio.loadmat(os.path.join(data_path, mat_file))
        label = np.load(os.path.join(label_path,'label.npy'))
        
        return data.transpose(2,0,1), label, data_path
