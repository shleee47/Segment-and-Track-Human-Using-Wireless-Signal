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
from torch.utils.data import DataLoader
import os
import pickle
from pathlib import Path
from trainer import ModelTrainer, ModelTester
from utils.setup import setup_solver
from utils.loss import create_criterion
from PANNs import ResNet38
from utils.utils import tr_val_split, test_split
from datasets import Audio_Reader, Audio_Collate, Test_Reader
from model import pretrained_Gated_CRNN8
from unet_model import UNet

def train(config):

    '''Dataset Preparation'''
    train_list, val_list, train_label, val_label = tr_val_split(config['datasets'])

    '''Dataloader'''
    train_dataset = Audio_Reader(train_list, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config['dataloader']['train']['batch_size'], shuffle=True, num_workers=config['dataloader']['train']['num_workers'])
    valid_dataset = Audio_Reader(val_list, val_label)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config['dataloader']['valid']['batch_size'], shuffle=True, num_workers=config['dataloader']['valid']['num_workers'])
    
    '''Model / Loss Criterion / Optimizer/ Scheduler'''
    model = UNet(150,1)
    criterion = create_criterion(config['criterion']['name'])
    optimizer, scheduler = setup_solver(model.parameters(), config)

    '''Trainer'''
    trainer = ModelTrainer(model, train_loader, valid_loader, criterion, optimizer, scheduler, config, **config['trainer'])
    trainer.train()


def test(config):

    '''Dataset Preparation'''
    test_list, test_label = test_split(config['datasets']['test'])

    '''Dataloader'''
    test_dataset = Test_Reader(test_list,test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=config['dataloader']['test']['num_workers'])

    '''Model'''
    model = UNet(150,1)

    '''Tester'''
    tester = ModelTester(model, test_loader, config['tester']['ckpt_path'], config['tester']['device'])
    tester.test()


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--mode', type=str, help='Train or Test')
    args = parser.parse_args()
    
    '''Load Config'''
    with open(os.path.join(args.config, args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    if args.mode == 'Train':
        train(config)
    elif args.mode == 'Test':
        test(config)
