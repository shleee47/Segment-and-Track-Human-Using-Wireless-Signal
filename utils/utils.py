from collections import namedtuple
import torch
from torch import nn
from utils.resnet import resnet18, resnet50, resnet101
import pdb
import random
import os
import csv
import numpy as np
import glob
import pickle
Encoder = namedtuple('Encoder', ('model', 'features', 'features_shape'))

def tr_val_split(config):

    ## Parameter Initialization
    split_ratio = 0.9
    seed_num = 100
    random.seed(seed_num)
    data_path = config['main']
    
    ## Data Preparation
    train_label_list, val_label_list = [],[]
    train_data_list,val_data_list = [],[]
    
    folder_list = [ f for f in os.listdir(os.path.join(data_path,'data'))]
    data_list = []
    label_list = []
    for folder_name in folder_list:
        data = os.path.join(data_path,'data',folder_name)
        label = os.path.join(data_path,'label',folder_name)
        data_list.append(data)
        label_list.append(label)

    temp = list(zip(data_list, label_list))
    random.shuffle(temp)
    data_list, label_list = zip(*temp)
    data_list, label_list = list(data_list), list(label_list)
    data_len = len(data_list)
    bnd_indx = int(data_len * split_ratio)

    ## data split
    train_data_list = data_list[:bnd_indx]
    val_data_list = data_list[bnd_indx:]

    ##label split 
    train_label_list = label_list[:bnd_indx] 
    val_label_list = label_list[bnd_indx:] 

    return train_data_list, val_data_list, train_label_list, val_label_list


def test_split(data_csv):

    csv_path = data_csv
    #pdb.set_trace()
    total_list = []
    csv_file = os.path.join(csv_path,'valid_mixed.csv')
    #with open(csv_file, newline='',encoding='cp949') as f:
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        tmp = list(reader)
        total_list += tmp
        f.close()


    #pdb.set_trace()
    test_label = []
    test_list = []
    for data in total_list:
        data = data[0]
        class_name = data.split('/')[5]
        data_path = data.strip('\n')
        test_label.append(class_name)
        test_list.append(data_path)

    #pdb.set_trace()
    return test_list, test_label


def intermediate_at_measures(n_class,encoded_ref,encoded_est):
    pdb.set_trace()
    #encoded_ref = torch.nn.functional.one_hot(encoded_ref,num_classes=n_class).detach().cpu().numpy()
    encoded_ref = encoded_ref.float().detach().cpu().numpy()
    #encoded_est = torch.nn.functional.one_hot(encoded_est,num_classes=n_class).float().detach().cpu().numpy()
    encoded_est = encoded_est.float().detach().cpu().numpy()

    tp = (encoded_est + encoded_ref == 2).sum()#axis=0)
    fp = (encoded_est - encoded_ref == 1).sum()#axis=0)
    fn = (encoded_ref - encoded_est == 1).sum()#axis=0)
    tn = (encoded_est + encoded_ref == 0).sum()#axis=0)

    return tp,fp,fn,tn

def make_npzeros(n_class):
    tp = np.zeros(n_class)
    tn = np.zeros(n_class)
    fp = np.zeros(n_class)
    fn = np.zeros(n_class)
    return tp,tn,fp,fn    

def calculate_f1_score(n_class,tp,fp,fn,tn):
    macro_f_measure = np.zeros(n_class)
    mask_f_score = 2 * tp + fp +fn != 0
    macro_f_measure[mask_f_score] = 2 * tp[mask_f_score] / (2 * tp + fp + fn)[mask_f_score]
    return macro_f_measure