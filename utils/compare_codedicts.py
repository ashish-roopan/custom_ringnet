import pickle
import numpy as np
import torch
import os
import cv2
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import models

sys.path.insert(0, '/home/ashish/code/custom_ringnet/')
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
from utils.debug_disp import Debug_diplay
from configs.config import cfg 






deca_dir = '/home/ashish/code/3D/DECA/'


#load deca input
with open(deca_dir + 'images.pkl', 'rb') as f:
	deca_input = pickle.load(f)

#load deca codedict
with open(deca_dir + 'codedict.pkl', 'rb') as f:
	deca_codedict = pickle.load(f)

#load deca params
with open(deca_dir + 'parameters.pkl', 'rb') as f:
	deca_params = pickle.load(f)

#load rignet input
with open('input.pkl', 'rb') as f:
	ringnet_input = pickle.load(f)
#load rignet codedict
with open('codedict.pkl', 'rb') as f:
	rignet_codedict = pickle.load(f)

with open('parameters.pkl', 'rb') as f:
	rignet_params = pickle.load(f)


print('deca input shape: ', deca_input.shape)
print('rignet input shape: ', ringnet_input.shape)
print()
print('deca params shape: ', deca_params.shape)
print('rignet params shape: ', rignet_params.shape)




device = torch.device('cuda:0')
n_param = 236
n_detail = 128
E_flame = ResnetEncoder(outsize=n_param).to(device) 
E_detail = ResnetEncoder(outsize=n_detail).to(device)

# resume model
model_path = cfg.pretrained_modelpath
if os.path.exists(model_path):
	print(f'trained model found. load {model_path}')
	checkpoint = torch.load(model_path)
	copy_state_dict(E_flame.state_dict(), checkpoint['E_flame'])
	copy_state_dict(E_detail.state_dict(), checkpoint['E_detail'])
else:
	print(f'please check model path: {model_path}')
# eval mode
E_flame.eval()
   
#run model    
with torch.no_grad():
	parameters = E_flame(deca_input)


print('deca params: ', deca_params[0:10])
print('rignet params: ', parameters[0:10])
print('deca params sum: ', torch.sum(deca_params))
print('rignet params sum: ', torch.sum(parameters))
print()
