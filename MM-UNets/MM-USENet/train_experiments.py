import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *



from utils.dataloader import HEP2Dataset
#from utils.dataloader_online import HEP2Dataset ## DECOMMENT IF YOU WANT TO EXTRACT RANDOM PATCHES

from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import numpy as np
import os

# SET GPU DEVICE
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#SET DIRECTORIES CHECKPOINT
dir_checkpoint_1 = Path('./checkpoints1/')
dir_checkpoint_2 = Path('./checkpoints2/')
dir_checkpoint_3 = Path('./checkpoints3/')
dir_checkpoint_4 = Path('./checkpoints4/')
dir_checkpoint_5 = Path('./checkpoints5/')
checkpoints_directories=[dir_checkpoint_1,dir_checkpoint_2,dir_checkpoint_3,dir_checkpoint_4,dir_checkpoint_5]

dir_imgs='/mnt/sdc1/llaudato6/dataset/HEp-2_dataset/data'

csv_dir_train_1="/mnt/sdc1/llaudato6/experiment1/exp1_train.csv"
csv_dir_test_1="/mnt/sdc1/llaudato6/experiment1/exp1_val.csv"

csv_dir_train_2="/mnt/sdc1/llaudato6/experiment2/exp2_train.csv"
csv_dir_test_2="/mnt/sdc1/llaudato6/experiment2/exp2_val.csv"

csv_dir_train_3="/mnt/sdc1/llaudato6/experiment3/exp3_train.csv"
csv_dir_test_3="/mnt/sdc1/llaudato6/experiment3/exp3_val.csv"

csv_dir_train_4="/mnt/sdc1/llaudato6/experiment4/exp4_train.csv"
csv_dir_test_4="/mnt/sdc1/llaudato6/experiment4/exp4_val.csv"

csv_dir_train_5="/mnt/sdc1/llaudato6/experiment5/exp5_train.csv"
csv_dir_test_5="/mnt/sdc1/llaudato6/experiment5/exp5_val.csv"

csv_directories_train=[csv_dir_train_1,csv_dir_train_2,csv_dir_train_3,csv_dir_train_4,csv_dir_train_5]
csv_directories_test=[csv_dir_test_1,csv_dir_test_2,csv_dir_test_3,csv_dir_test_4,csv_dir_test_5]

def train_net(
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              save_checkpoint: bool = True,
              amp: bool = False,
              load_data: int=0,
              save_data: int=0):
    f=open("RESULTS_EXPERIMENTS.txt","w")
    for k in range(5):
        f.write("--------------------------------------------------\n")
        f.write("FOLD "+ str(k+1)+"\n")
        f.write("--------------------------------------------------\n")
        print("---------")
        print("FOLD ", str(k+1))
        if load_data==0:
            # 1. Create dataset
            train_set = HEP2Dataset(csv_directories_train[k], dir_imgs)
            test_set=HEP2Dataset(csv_directories_test[k], dir_imgs)
            # 3. Create data loaders
            train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)

            if save_data==1:
                torch.save(train_loader,"/mnt/sdc1/llaudato6/experiment"+str(k+1)+"/train_loader.pth")
                torch.save(test_loader,"/mnt/sdc1/llaudato6/experiment"+str(k+1)+"/val_loader.pth")
                print("DATALOADERS SAVED!")
                

        elif load_data==1:
            train_loader=torch.load("/mnt/sdc1/llaudato6/experiment"+str(k+1)+"/train_loader.pth")
            test_loader=torch.load("/mnt/sdc1/llaudato6/experiment"+str(k+1)+"/val_loader.pth")
            print("DATALOADERS IMPORTED!")
        
        n_train=len(train_loader)*batch_size
        n_test=len(test_loader)*batch_size
        print("LEN_TRAINING: ", n_train)
        print("LEN_TEST: ", n_test)

        #CREATE NET MODEL
        net = UNet(n_channels=1, n_classes=8, bilinear=False)
        logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
        net.to(device=device)
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-5, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)  # goal: minimize loss
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        
        ##CREATE BLACK MASK TO APPEND TO OUTPUT MASK TO REBUILD THE ARRAY OF MASKS
        black_mask=np.zeros((384,384))
        black_mask=torch.as_tensor(black_mask, dtype=torch.float32)

        #define CrossEntropyLosses e Accuracy
        criterion_train = nn.CrossEntropyLoss()
        criterion_test = nn.CrossEntropyLoss()


        global_step = 0


        # 5. Begin training
        for epoch in range(24, epochs+1):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for i,item in enumerate(train_loader):
                    image= item['image']/255
                    true_masks = item['mask']

                    ##TRANSFER TO GPU DEVICE 
                    image = image.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.float32)
                    assert image.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {image.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                    
                    with torch.cuda.amp.autocast(enabled=amp):
                        pred_masks, _= net(image) 
                        probs=F.softmax(pred_masks, dim=1)

                        #COMPUTE LOSSES TO BACKPROPAGATE
                        ce_seg=criterion_train(probs.cpu(),true_masks.cpu())
                        dl_seg=dice_loss(probs.cpu(), true_masks.cpu(), multiclass=True)
                        loss=dl_seg + ce_seg ###comment if you no want to propagate ce_loss on segmentation task
                    
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(image.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = (n_train // (batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            test_loss_mean = evaluate(batch_size,net, test_loader, device, criterion_test)
                            scheduler.step(test_loss_mean)
                            logging.info('Test Cross Entropy: {}'.format(test_loss_mean))
        
            if save_checkpoint:
                Path(checkpoints_directories[k]).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(checkpoints_directories[k] / 'checkpoint_epoch{}_{}.pth'.format(epoch,epoch_loss/len(train_loader))))
                logging.info(f'Checkpoint {epoch} saved!')
    f.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--load_data', type=int, default=1, help='Load dataloaders from a .pth file')
    parser.add_argument('--save_data', type=int, default=0, help='Save dataloaders into a .pth file')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=8, help='Number of classes')
    parser.add_argument('--channels', '-n', type=int, default=1, help='Number of input img channels')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    try:
        train_net(epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  load_data=args.load_data,
                  save_data=args.save_data)
    except KeyboardInterrupt:
        raise
    