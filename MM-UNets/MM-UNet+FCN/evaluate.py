import torch
import torch.nn.functional as F
from tqdm import tqdm


from utils.dice_score import dice_loss

def evaluate(batch_size,net, dataloader, device, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    loss=0
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches*batch_size, desc='Validation round', unit='batch', leave=False):
        image, true_masks,label = batch['image']/255, batch['mask'], batch['label']
        label=label.to(device=device,dtype=torch.int16)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)
        output, lab_pred_embedding = net(image)
        lab_pred_embedding=lab_pred_embedding.to(device=device,dtype=torch.float32)
        label=F.one_hot(label.to(torch.int64),7)
        label=label.to(device=device,dtype=torch.float32)
        probs= F.softmax(output, dim=1)#SOFTMAX
        with torch.no_grad():
            ce_seg=criterion(probs.cpu(),true_masks.cpu())
            dl_seg=dice_loss(probs.cpu(), true_masks.cpu(), multiclass=True)
            ce_pat=criterion(lab_pred_embedding.cpu(), label.cpu())
            mean_loss=dl_seg+ce_pat #+ce_seg ###decomment if you want to propagate ce_loss on segmentation task
            loss += mean_loss.item()
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return loss
    return loss/ num_val_batches
