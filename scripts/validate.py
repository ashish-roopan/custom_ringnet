import torch


def validate_epoch(model, dataloader, criteria, device,  wandb=None):
    valid_loss = 0.0
    model.eval()
    for i, (full_image, inputs, labels) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # Compute loss
            loss = criteria(outputs, labels)
            # diff = outputs - labels
            # loss = torch.norm(diff, dim=1, p=2).square().mean()
            valid_loss += loss.item() * inputs.size(0)

            #wandb logging
            if wandb:
                wandb.log({'valid_loss': loss.item()})

    
    return valid_loss