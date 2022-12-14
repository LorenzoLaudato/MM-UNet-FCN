import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.nn as nn


from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss

def evaluate(batch_size,net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    loss=0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches*batch_size, desc='Validation round', unit='batch', leave=False):
        image, true_masks,label = batch['image']/255, batch['mask'], batch['label']
        true_masks[true_masks>1.0]=1.0
        label=label.to(device=device,dtype=torch.int16)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        output, _ = net(image)
        probs= F.softmax(output, dim=1)#SOFTMAX
        with torch.no_grad():
            mean_loss=criterion(probs.cpu(),true_masks.cpu())+dice_loss(probs.cpu(),true_masks.cpu(), multiclass=True)
            loss += mean_loss.item()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return loss
    return loss/ num_val_batches
