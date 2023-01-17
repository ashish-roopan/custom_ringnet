import torch
import torch.nn as nn

def train_epoch(model, decoder, optimizer, dataloader,criterion, scheduler, device, wandb):
    train_loss = 0.0
    model.train()
    
    for i, (image, gt_landmarks) in enumerate(dataloader):
        inputs = image.to(device)
        gt_landmarks = gt_landmarks.to(device)

        #forward pass            
        pred_cam, pred_pose, pred_shape, pred_exp = model(inputs)

        #decode model output
        vertices, projected_landmarks = decoder.decode(pred_shape, pred_pose, pred_exp, pred_cam)
        

        # Compute loss
        
        lmk_loss = criterion(projected_landmarks[:,:60], gt_landmarks[:,:60]) * 50
        
        shape_reg_loss = nn.MSELoss()(pred_shape, torch.zeros_like(pred_shape))  *0.01
        exp_reg_loss = nn.MSELoss()(pred_exp, torch.zeros_like(pred_exp)) *0.01
        pose_reg_loss = nn.MSELoss()(pred_pose, torch.zeros_like(pred_pose)) *0.01
        reg_loss = shape_reg_loss + exp_reg_loss + pose_reg_loss

        loss = lmk_loss + reg_loss
        
        # Backward pass
        train_loss += loss.item() * inputs.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        wandb.log({'train_loss': loss.item()})
    return train_loss






       
    