import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import argparse
import torch
import torch.nn as nn
import pdb
import yaml 
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import os
import pickle
from pathlib import Path
import json
import glob
import torchaudio

'''Feature Extractor'''
class Feature_Extractor():
    def __init__(self, n_fft=512, hopsize=128, window='hann'):
        self.nfft = n_fft
        self.hopsize = 128
        self.window = 'hann'
        self.melW = 128
        self.n_mfcc= 40

    def MFCC(self, sig): 
        S = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=self.n_mfcc,
                n_fft = self.nfft, hop_length = self.hopsize)

        return S

'''Tester'''
class ModelTester:
    def __init__(self, model, ckpt_path, device):

        # Essential parts
        self.device = torch.device('cuda:{}'.format(device))
        #self.model = model.to(self.device)
        self.model = model.cuda()
        self.load_checkpoint(ckpt_path)
        self.model.eval()

    def load_checkpoint(self, ckpt):
        print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        

    def test(self,batch):

        with torch.no_grad():
            pdb.set_trace()
            inputs = batch
            inputs = inputs.cuda()
            outputs = self.model(inputs)
            scores = outputs['clipwise_output']
            best_prediction = scores.max(-1)[1]
            #################################

            return best_prediction



'''Network Fintuning'''
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer,'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class finetunePANNs(nn.Module):
    def __init__(self,PANNs_pretrain,class_num):
        super(finetunePANNs, self).__init__()
        self.PANNs = PANNs_pretrain

        self.add_fc1 = nn.Linear(527,class_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.add_fc1)

    def forward(self, input):
        x=  self.PANNs(input)
        embed = x['embedding']
        clipwise_output = torch.sigmoid(self.add_fc1(embed))
        output_dict = {'clipwise_output': clipwise_output}

        return output_dict

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x


'''Network'''
def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class ResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, 
        classes_num):
        super(ResNet38, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.bn0 = nn.BatchNorm2d(40)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)
        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)
        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)


    def forward(self, input):
        """
        Input: (batch_size, T, F)"""
        
        x= input.transpose(1,3)
        #x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        embedding = self.fc_audioset(embedding)

        output_dict = {'embedding': embedding}

        return output_dict


'''JSON'''
def sp2clock(idx, sr=16000):
    sec = int(idx/16000)
    minu = int(sec//60)
    result = '{:02d}:{:02d}'.format(minu, sec-60*minu)
    return result

def make_NONE(json_output):
    for s in json_output['task2_answer'][0]:
        for idx, d in enumerate(json_output['task2_answer'][0][s]):
            for k, v in d.items():
                for classes in json_output['task2_answer'][0][s][idx][k]:
                    for c in classes:
                        if not json_output['task2_answer'][0][s][idx][k][0][c]:
                            json_output['task2_answer'][0][s][idx][k][0][c] = ['NONE']
    return json_output


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    '''init answer json file'''
    json_output_path = '/root/VARCO/answersheet_3_00_VARCO.json'

    # define output format
    json_output = {
            'task2_answer': [{
                    'set_1':[
                            {'drone_1': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_2': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_3': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]}],
                    'set_2':[
                            {'drone_1': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_2': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_3': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]}],
                    'set_3':[
                            {'drone_1': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_2': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_3': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]}],
                    'set_4':[
                            {'drone_1': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_2': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_3': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]}],
                    'set_5':[
                            {'drone_1': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_2': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]},
                            {'drone_3': [
                                    {'M': [],
                                    'W': [],
                                    'C': []}]}]
            }]
        }

    '''Feature Extractor Init'''
    f_extractor = Feature_Extractor()

    '''Model Init'''
    PANNs_model = ResNet38(32000, 1024, 1024//4,64, 3)
    gender_model = finetunePANNs(PANNs_model,3)

    '''Tester Init'''
    tester = ModelTester(gender_model, '../model_ckpt/gender_best.pt', 0)

    '''Load Pickle'''
    pkl_path = '/home/agc2021/temp/pickle/2_vad/'
    pickle_paths = sorted(glob.glob(pkl_path, '*.pkl')) # num: 15

    '''Main'''
    pdb.set_trace()
    for pickle_path in pickle_paths:
        # load pickle, wav
        with open(pickle_path, 'rb') as p:
            pickle_data = pickle.load(p)

        wav_path = pickle_data['output_path']
        set_name, drone_name, _ = wav_path.split('/')[-1].split('_')
        set_num, drone_num = int(set_name[-1]), int(drone_name[-1])
        set_name = set_name[:3] + '_' + str(set_num)
        drone_name = drone_name[:5] + '_' + str(drone_num)

        wav, sr = librosa.load(wav_path) # sr=16000

        # get VAD timestamps
        timestamps = pickle_data['time']

        pdb.set_trace()
        # iter and load wav, pass gender classification
        for ts in timestamps:
            start_ts, end_ts = ts
            start_idx, end_idx = int(start_ts*sr), int(end_ts*sr)
            duration = end_idx-start_idx
            wav_chunk = wav[:,start_idx:end_idx]

            '''Feature Extraction'''
            feature = f_extractor.MFCC(wav_chunk)
            feature = np.expand_dims(feature,axis=0)
            m_inputs =  torch.FloatTensor(feature).transpose(1,2)

            '''Model Forwarding'''
            m_outputs = tester.test(m_inputs)
            pdb.set_trace()

            if m_outputs == 0:
                output = {'M': None, 'W': None, 'C': duration/2}
            elif m_outputs == 1:
                output = {'M': None, 'W': duration/2, 'C': None}
            elif m_outputs == 2:
                output = {'M': duration/2, 'W': None, 'C': None}

            output = {'M': 1*sr, 'W': None, 'C': 2*sr} # model output example

            # append to json_output
            for class_type, class_sp_idx in output.items():
                if class_sp_idx:
                    json_output['task2_answer'][0][set_name][drone_num-1][drone_name][0][class_type].append(sp2clock(start_idx+class_sp_idx))

    # save json
    with open(json_output_path, 'w') as j:
        data = json.dumps(make_NONE(json_output), indent=4)
        j.write(data)