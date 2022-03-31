import os
import sys
import time
import numpy as np
import datetime
import pickle as pkl
from pathlib import Path
import torch
import pdb
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from utils.utils import intermediate_at_measures, make_npzeros, calculate_f1_score
import logging
import json
from multiprocessing import Pool
import time
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, config, epochs, device, save_path, ckpt_path=None, comment=None, fold=2):

        self.device = torch.device('cuda:{}'.format(device))
        #self.model = model.cuda()
        self.model = model
        #self.n_class = config['MYNET']['n_classes']
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.exp_path = Path(os.path.join(save_path, datetime.now().strftime('%d%B_%0l%0M'))) #21November_0430
        self.exp_path.mkdir(exist_ok=True, parents=True)

        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        
        #Dump hyper-parameters
        with open(str(self.exp_path.joinpath('config.json')), 'w') as f:
            json.dump(config, f, indent=2)

        if comment != None:
            self.logger.info(comment)

        self.writter = SummaryWriter(self.exp_path.joinpath('logs'))
        self.epochs = epochs
        self.best_loss = 100.0
        self.best_epoch = 0
        self.eps = 1e-08
        
        if ckpt_path != None:
            self.load_checkpoint(ckpt_path)
            self.optimizer.param_groups[0]['lr'] = 0.0001

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            train_loss = self.train_single_epoch(epoch)
            valid_loss = self.inference()
            duration = time.time() - start

            if v_loss < self.best_acc:
                self.best_loss = v_loss
                self.best_epoch = epoch

            self.scheduler.step(v_loss)
            # self.logger.info("epoch: {} --- t_loss : {:0.3f}, train_acc = {}%, v_loss: {:0.3f}, val_acc: {}%, \
            #                     best_acc: {}%, best_epoch: {}, time: {:0.2f}s, lr: {}"\
            #                     .format(epoch, train_loss, t_accuracy, valid_loss, v_accuracy,\
            #                     self.best_acc, self.best_epoch, duration,self.optimizer.param_groups[0]['lr']))
            self.logger.info("epoch: {} --- t_loss : {:0.3f}, v_loss: {:0.3f}, \
                                 best_loss: {}, best_epoch: {}, time: {:0.2f}s, lr: {}"\
                                .format(epoch, train_loss, valid_loss, \
                                self.best_loss, self.best_epoch, duration,self.optimizer.param_groups[0]['lr']))

            self.save_checkpoint(epoch, v_loss)

            ## Update Tensorboard 
            self.writter.add_scalar('data/Train/Loss', train_loss, epoch)
            self.writter.add_scalar('data/Valid/Loss', valid_loss, epoch)
            # self.writter.add_scalar('data/Train/Accuracy', t_accuracy, epoch)
            # self.writter.add_scalar('data/Valid/Accuracy', v_accuracy, epoch)

        self.writter.close()


    def train_single_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0
        total_data_num = 0
        batch_size = len(self.train_loader)

        #pdb.set_trace()
        for b, batch in (enumerate(self.train_loader)):
            inputs, labels = batch
            B, C, T, FT = inputs.size()
            #inputs = inputs.cuda()
            #labels = labels.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(inputs.float())
            #pdb.set_trace()

            preds = outputs > 0.5
            total_data_num += B

            batch_loss = self.criterion(outputs.squeeze(1), labels.float())
            batch_loss.backward()
            total_loss += batch_loss.item()
            self.optimizer.step()

            #print("{}/{}: {}/{}".format(b, batch_size, correct_cnt, tot_cnt), end='\r')
        #pdb.set_trace()
        mean_loss = total_loss / (total_data_num+self.eps)
        return mean_loss#, (correct_cnt/tot_cnt)*100


    def inference(self):
        self.model.eval()
        
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0
        total_data_num = 0
        batch_size = len(self.valid_loader)


        with torch.no_grad():
            for b, batch in enumerate(self.valid_loader):
                #pdb.set_trace()
                inputs, labels, data_len = batch
                B, C, T, Ft = inputs.size()
                inputs = inputs.cuda()
                labels = labels.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())

                preds = outputs > 0.5
                total_data_num += B

                batch_loss = self.criterion(outputs.squeeze(1), labels.float())
                total_loss += batch_loss.item()

                #print("{}/{}: {}/{}".format(b, batch_size, correct_cnt, tot_cnt), end='\r')

            mean_loss = total_loss / total_data_num
            return mean_loss#, (correct_cnt/tot_cnt)*100


    def load_checkpoint(self, ckpt):
        self.logger.info("Loading checkpoint from {ckpt}")
        print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer'])#, strict=False)


    def save_checkpoint(self, epoch, vacc, best=True):
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        self.exp_path.joinpath('ckpt').mkdir(exist_ok=True, parents=True)
        save_path = "{}/ckpt/{}_{:0.4f}.pt".format(self.exp_path, epoch, vacc)
        torch.save(state_dict, save_path)


class ModelTester:
    def __init__(self, model, test_loader, ckpt_path, device):

        # Essential parts
        self.device = torch.device('cuda:{}'.format(device))
        self.model = model.cuda()
        self.test_loader = test_loader

        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(sh)

        self.load_checkpoint(ckpt_path)
        self.class_list = ['Child','Female','Male']

    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        # print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
    def test(self):
        """
        images : [B x T x C x H x W]
        labels : [B x T]
        """
        self.model.eval()
        batch_size = len(self.test_loader)
        
        with torch.no_grad():
            for b, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                pdb.set_trace()
                inputs, labels, data_path = batch
                B, C, T, Ft = inputs.size()  
                inputs = inputs.cuda()
                outputs = self.model(inputs)

                print("instance segmentation results: {}".format(outputs))