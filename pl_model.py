
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

import os
import logging
import json
import math
import torch
import datetime
import argparse
import torch
import dataloaders
import models
import inspect
import math
import losses
import numpy as np
import random
from utils import Logger
from utils.torchsummary import summary
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
import utils

class pl_model(pl.LightningModule):

    def __init__(self, model, loss, hparams):
        super().__init__()
        
        #self.save_hyperparameters(**hparams)
        self.hparams = hparams
        #print(self.hparams)
        
        self.num_classes = hparams['num_classes']
        self.model = model
        self.loss = loss
        
        self._reset_metrics()

  

    def forward(self, x):
        return self.model(x)
               
    
    def on_epoch_start(self):
        self._reset_metrics()
        
        optimizer = self.optimizers()

        print('\n')
        print(f'Epoch: {self.current_epoch}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    
    def on_pre_performance_check(self):
        self._reset_metrics()
        print('\n###### EVALUATION ######')



    def training_step(self, batch, batch_idx):

        data, target = batch
        output = self.model(data)
        loss = self.loss(output, target)

        self.total_loss.update(loss.item())
        
        
        seg_metrics = eval_metrics(output, target, self.num_classes)
        self._update_seg_metrics(*seg_metrics)
        pixAcc, mIoU, _ = self._get_seg_metrics().values()
        
        log_str = 'TRAIN ({}) | Loss: {:.3f} | Acc {:.2f} mIoU {:.2f} |'.format(self.current_epoch, self.total_loss.average, pixAcc, mIoU)
        self.log('train_log', log_str, prog_bar=True) 
        
        return loss
      
      
    def training_epoch_end(self, step_loss):
        seg_metrics = self._get_seg_metrics()
        log = {'loss': self.total_loss.average, **seg_metrics}
        log_str = ''
        for k, v in log.items():
            log_str += k+'：'+v+'\n'
        self.log('train_epoch_log', log_str, prog_bar=False)
               
        return log
        

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.loss(output, target)
        self.total_loss.update(loss.item())
        seg_metrics = eval_metrics(output, target, self.num_classes)
        self._update_seg_metrics(*seg_metrics)
        
        pixAcc, mIoU, _ = self._get_seg_metrics().values()
        
        log_str = 'EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( self.current_epoch, self.total_loss.average,pixAcc, mIoU)
        self.log('eval_log', log_str, prog_bar=True)
        
        return loss

    def validation_epoch_end(self, outputs):
        seg_metrics = self._get_seg_metrics()
        
        log = {'val_loss': self.total_loss.average, **seg_metrics}
        log_str=''
        for k, v in log.items():
            log_str += k + ':'+v+'\n'
        self.log('val_epoch_log', log_str, prog_bar=False)

        return log



    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams['optimizer']['args']['lr'])
      
        #scheduler =  getattr(torch.optim.lr_scheduler, self.hparams['lr_scheduler']['type'])(optim, **self.hparams['lr_scheduler']['args'])
        scheduler = getattr(utils.lr_scheduler, self.hparams['lr_scheduler']['type'])(optim, **self.hparams['lr_scheduler']['args'])

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optim, T_max=self.hparams.T_max, eta_min=1e-6, last_epoch=-1, verbose=False
        #)
        return [optim], [scheduler]
 
    
    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

    
        
        