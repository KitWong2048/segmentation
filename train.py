import os
import json
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


import pytorch_lightning as pl
from pl_model import pl_model
from dataloaders.rsi import *

seed = 1000000
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, arg):

    train_logger = Logger()

    # DATA LOADERS
    print("Loading train data...")
    train_loader = get_RSI_loader( **config['train_loader']['args']  )
    
    #train_loader = get_instance(dataloaders, 'train_loader', config)
    print("Loading val data...")
    val_loader = get_RSI_loader( **config['val_loader']['args'] )
    #val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    net = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    # print(f'\n{model}\n')

    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    #setattr(config, 'num_classes', train_loader.dataset.num_classes)
    #print(train_loader.dataset.num_classes)
    
    config['num_classes'] = train_loader.dataset.num_classes
    
    model = pl_model(net, loss, config)
    
    
    if arg.resume:
        model.load_from_checkpoint(arg.resume)
    
    
    #trainer = pl.Trainer(default_root_dir=arg.checkpoints_folder, gpus="0")
    #trainer = pl.Trainer(default_root_dir=arg.checkpoints_folder, tpu_cores=8)
    trainer = pl.Trainer(default_root_dir=arg.checkpoints_folder, tpu_cores=[1])

    
    trainer.fit(model=model,  train_dataloader=train_loader, val_dataloaders=val_loader )
    
    
    

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--checkpoints_folder', default='/content/drive/My Drive/segmentation/checkpoints', type=str,
                        help='folders for storing checkpoints')
                        
    args = parser.parse_args()

    config = json.load(open(args.config))
    # if args.resume:
    #     config = torch.load(args.resume)['config']
    #if args.device:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args)