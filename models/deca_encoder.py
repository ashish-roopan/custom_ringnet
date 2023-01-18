# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import resnet

import pickle
import sys
sys.path.append('../')
from configs.config import cfg

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import resnet

# class ResnetEncoder(nn.Module):
#     def __init__(self, outsize, last_op=None):
#         super(ResnetEncoder, self).__init__()
#         feature_size = 2048
#         self.encoder = resnet.load_ResNet50Model() #out: 2048
#         ### regressor
#         self.layers = nn.Sequential(
#             nn.Linear(feature_size, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, outsize)
#         )
#         self.last_op = last_op

#     def forward(self, inputs):
#         features = self.encoder(inputs)
#         parameters = self.layers(features)
#         if self.last_op:
#             parameters = self.last_op(parameters)
#         return parameters




class ResnetEncoder(nn.Module):
    def __init__(self, outsize, last_op=None):
        super(ResnetEncoder, self).__init__()
        feature_size = 2048
        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )
        self.param_dict = {i:cfg.model.get('n_' + i) for i in cfg.model.param_list}

    def forward(self, inputs):
        features = self.encoder(inputs)
        parameters = self.layers(features)
        codedict = self.decompose_code(parameters, self.param_dict)
            
        return codedict['cam'], codedict['pose'], codedict['shape'], codedict['exp']

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict