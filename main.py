import os
import cv2
import random
import numpy as np
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import models

from models.fc import Fc
from models.Resnet_encoder import Encoder
from models.deca_encoder import ResnetEncoder
from models.FLAME import FLAME
from utils.helper import Helper
from utils.render import Renderer
from utils.util import copy_state_dict
from models.decoder import Decoder
from dataloaders.dataloader import get_dataloader
from dataloaders.deca_dataloader import get_dataloader
from scripts.train import train_epoch
from scripts.validate import valid_epoch
from utils.debug_disp import Debug_diplay
from configs.config import cfg 
from losses.adaptive_wing_loss import AdaptiveWingLoss

import wandb
wandb.init(project="cloth_project")

def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)
    cudnn.deterministic = True
    cudnn.benchmark = False
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/flicker/', help='data directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--save_model', type=bool, default=False, help='save model')
    parser.add_argument('--load_model', type=bool, default=False, help='load_model')
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pt', help='model path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--debug', type=bool, default=False, help='debug')
    parser.add_argument('--num_images', type=int, default=3200, help='number of images to train')
    return parser.parse_args()


#.            Initialize                     
args = parse_args()
deterministic(args.seed)
Render = Renderer(args.device)
helper = Helper(args.device)
flame = FLAME(cfg.model).to(args.device)

renderer = Render.create_render()
debug = Debug_diplay(args.device, flame, helper, renderer)
debug_disp = debug.debug_disp

#.               Prepare Data                 
data_dir = args.data_dir
train_dataloader = get_dataloader(data_dir , batch_size=args.batch_size, num_images=args.num_images, split='train')
# valid_dataloader = get_dataloader(data_dir + 'valid_set/', batch_size=args.batch_size, split='valid')
test_dataloader = get_dataloader(data_dir + 'test_set/', batch_size=args.batch_size, split='test')

#.           Prepare Model                    
# model = Encoder()
n_params = 236
model = ResnetEncoder(outsize=n_params) 

# resume model
model_path = cfg.pretrained_modelpath
if os.path.exists(model_path):
    print(f'trained model found. load {model_path}')
    checkpoint = torch.load(model_path)
    print(checkpoint.keys())
    copy_state_dict(model.state_dict(), checkpoint['E_flame'])
   
decoder = Decoder(device=args.device)
model = model.to(args.device)

#.          Hyperparameters                   
start_epoch = 0
best_valid_loss = 0.104
lr = 0.00001
momentum = 0.9
weight_decay = 0.0
epochs = args.num_epochs
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.000001, steps_per_epoch=len(train_dataloader), epochs=epochs)
# scheduler = scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600, 900, 1500, 2000], gamma=0.1)
scheduler = None
# criteria = torch.nn.SmoothL1Loss()
# criteria = torch.nn.MSELoss()
criteria = AdaptiveWingLoss()


if args.load_model:
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = checkpoint['lr_sched']
    start_epoch = checkpoint['epoch']
    best_valid_loss = checkpoint['best_valid_loss']
    best_valid_loss = 0.104
    lr = scheduler.get_last_lr()

wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": args.batch_size,
    "weight_decay" : 0.01,
    "momentum" : 0.9,
    "optimizer" : optimizer,
    "scheduler" : scheduler
}

#.                      Training Loop                       
for epoch in range(epochs):
    train_loss = train_epoch(model, decoder, optimizer, train_dataloader, criteria, scheduler, args.device, wandb)
    test_loss = valid_epoch(model, decoder, test_dataloader, criteria, args.device, wandb)

    avg_train_loss = train_loss / len(train_dataloader)
    avg_test_loss = test_loss / len(test_dataloader)
    print(f'Epoch:{start_epoch + epoch} Training Loss:{avg_train_loss} Test Loss:{avg_test_loss}')

    ## debug display
    if args.debug and epoch % 1 == 0:
        try:
            train_debug_disp = debug_disp(model, train_dataloader)
            cv2.imshow('train', train_debug_disp)
            test_debug_disp = debug_disp(model, test_dataloader)
            cv2.imshow('test', test_debug_disp)
        except:
            cv2.imshow('test', np.zeros((256, 256, 3), dtype=np.uint8))
            cv2.imshow('train', np.zeros((256, 256, 3), dtype=np.uint8))
        cv2.waitKey(1)

    