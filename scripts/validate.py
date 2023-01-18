import torch
import torch.nn as nn

def valid_epoch(model, decoder, dataloader, criterion, device,  wandb=None):
    valid_loss = 0.0
    model.eval()


    for i, (images, gt_landmarks) in enumerate(dataloader):
        with torch.no_grad():
            inputs = images.to(device)
            gt_landmarks = gt_landmarks.to(device)
            
            outputs = model(inputs)
            #forward pass            
            pred_cam, pred_pose, pred_shape, pred_exp = model(inputs)
    
            #decode model output
            vertices, projected_landmarks = decoder.decode(pred_shape, pred_pose, pred_exp, pred_cam)

            # Compute loss
            lmk_loss = criterion(projected_landmarks[:,:], gt_landmarks[:,:]) *1
            shape_reg_loss = nn.MSELoss()(pred_shape, torch.zeros_like(pred_shape))  *0.0001
            exp_reg_loss = nn.MSELoss()(pred_exp, torch.zeros_like(pred_exp)) *0.0001
            pose_reg_loss = nn.MSELoss()(pred_pose, torch.zeros_like(pred_pose)) *0.0001
            reg_loss = shape_reg_loss + exp_reg_loss + pose_reg_loss
            loss = lmk_loss 

            valid_loss += loss.item() * inputs.size(0)

            #wandb logging
            if wandb:
                wandb.log({'valid_loss': loss.item()})

    
    return valid_loss
