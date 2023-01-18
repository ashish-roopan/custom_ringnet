import torch
import torch.nn as nn
import sys
sys.path.append('../')

from losses import lossfunc
from configs.config import cfg

def train_epoch(model, decoder, optimizer, dataloader,criterion, scheduler, device, wandb):
    train_loss = 0.0
    model.train()
    
    for i, (image, gt_landmarks) in enumerate(dataloader):
        inputs = image.to(device)
        gt_landmarks = gt_landmarks.to(device)

        #forward pass            
        pred_cam, pred_pose, pred_shape, pred_exp = model(inputs)

        #decode model output
        vertices, predicted_landmarks = decoder.decode(pred_shape, pred_pose, pred_exp, pred_cam)
        

        # Compute loss
        weight = lossfunc.get_weight()

        landmark_loss = criterion(weight * predicted_landmarks[:,:], weight * gt_landmarks[:,:]) 
        eye_distance_loss = lossfunc.eyed_loss(predicted_landmarks, gt_landmarks) * cfg.loss.eyed
        lip_distance_loss = lossfunc.lipd_loss(predicted_landmarks, gt_landmarks) * cfg.loss.lipd
        shape_reg_loss = (torch.sum(pred_shape**2)/2) * cfg.loss.reg_shape
        expression_reg_loss = (torch.sum(pred_exp**2)/2) * cfg.loss.reg_exp

        loss = landmark_loss + eye_distance_loss + lip_distance_loss + shape_reg_loss + expression_reg_loss

        # Backward pass
        train_loss += loss.item() * inputs.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        wandb.log({'train_loss': loss.item()})
    return train_loss






       
    