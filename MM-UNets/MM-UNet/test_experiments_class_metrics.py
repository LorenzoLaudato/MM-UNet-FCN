import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchmetrics import Accuracy
from utils.utils import *
from sklearn.metrics import confusion_matrix
from kornia.morphology import opening, closing



from utils.dataloader import HEP2Dataset
from unet import UNet
import numpy as np
import os

## SET GPU DEVICE
os.environ["CUDA_VISIBLE_DEVICES"]="1"

## SET PATH OF OUTPUT IMAGES
dir_save_pred="/outputs/"

## SET PATH OF DATASET
dir_imgs='/mnt/sdc1/llaudato6/dataset/HEp-2_dataset/data'


def test_net(net,
              device,
              batch_size: int = 1,
              load_data: int=0,
              save_data: int=0,
              fold: int=0,
              visualize: int=0,
              mathematical_morph: int=0):
        print("FOLD ", str(fold))

        if load_data==0:
            #SET PATH IF YOU NO HAVE A PRE-SAVED DATALOADER (CSV PATH CONTAINING LABELS OF DATASET)
            test_set=HEP2Dataset("/mnt/sdc1/llaudato6/experiment"+str(fold)+"/exp"+str(fold)+"_val.csv", dir_imgs)
            # 3. Create data loaders
            test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size=batch_size, num_workers=4, pin_memory=True)

            if save_data==1:
                torch.save(test_loader,"/mnt/sdc1/llaudato6/experiment"+str(fold)+"/val_loader.pth")
                print("DATALOADER SAVED!")
                    

        elif load_data==1:
            test_loader=torch.load("/mnt/sdc1/llaudato6/experiment"+str(fold)+"/val_loader.pth")
            print("DATALOADER IMPORTED!")
            
        n_test=len(test_loader)*batch_size
        print("LEN_TEST: ", n_test)

        ##CREATE BLACK MASK TO APPEND TO OUTPUT MASK TO REBUILD THE ARRAY OF MASKS
        black_mask=np.zeros((384,384))
        black_mask=torch.as_tensor(black_mask, dtype=torch.float32)

        #ACCURACY FOR SEGMENTATION ACCURACY
        acc=Accuracy()

        # 5. Begin test
        net.eval()

        ##INITIALIZE ARRAYS FOR PREDICTION
        y_true=[]
        y_pred=[]
        specimen_true=[]
        specimen_pred=[]
        n_specimens=[0,0,0,0,0,0,0]
        seg_accuracies=[0,0,0,0,0,0,0]
        dice_scores=[0,0,0,0,0,0,0]

        for i, item in enumerate(test_loader):
            img = item['image']/255
            true_masks=item['mask']
            label=item['label']
            ##TRANSFER TO GPU DEVICE 
            img = img.to(device=device, dtype=torch.float32)
            

            with torch.no_grad():
                pred_masks,_ = net(img)
                probs = F.softmax(pred_masks, dim=1)
                one_hot=F.one_hot(probs.argmax(dim=1), net.n_classes).permute(0,3,1,2)
                output= (probs>0.45).float()
                if mathematical_morph:
                    ## COMPUTE MATHEMATIC MORPHOLOGICAL OPERATIONS, YOU CAN SET DIM OF THE KERNEL SIZE
                    kernel=torch.ones(7,7)
                    one_hot=opening(one_hot.cpu(),kernel.cpu())
                    one_hot=closing(one_hot.cpu(),kernel.cpu())

                for j in range(batch_size):
                    max_label,_=find_max_mask(probs[j,...])
                    y_true.append(label[j].item())
                    y_pred.append(max_label)
                    ###SUM ON 7 CHANNEL MASKS TO OBTAIN THE FINAL PREDICTED MASK
                    compact=torch.zeros(384,384)
                    for l in range(7):
                        compact+=one_hot[j,l,:,:].cpu()
                    specimen_true.append(true_masks[j,label[j],:,:])
                    specimen_pred.append(compact)
                    ## EACH 15 PATCHES, WE RECONSTRUCT THE SPECIMEN
                    if (i*batch_size + j+1)%15==0:
                        spec_true=patch_reconstruction(specimen_true)
                        spec_pred=patch_reconstruction(specimen_pred)
                        spec_true=np.asarray(spec_true)
                        spec_pred=np.asarray(spec_pred)
                        spec_pred=torch.as_tensor(spec_pred, dtype=torch.float32)
                        spec_true=torch.as_tensor(spec_true, dtype=torch.float32)
                        specimen_true=[]
                        specimen_pred=[]
                        if visualize==1:
                            save_image(spec_true,dir_save_pred+"/SPECIMENS/SPEC_TRUE"+str(i*batch_size + j+1)+".png")
                            save_image(spec_pred,dir_save_pred+"/SPECIMENS/SPEC_PRED"+str(i*batch_size + j+1)+".png")
                        dice_scores,seg_accuracies,n_specimens=update_seg_accuracies_and_dice_scores(dice_scores,seg_accuracies,n_specimens,spec_pred,spec_true, label[j].item(), acc)

                    if visualize==1:
                        save_image(img[j,0,:,:].float(),dir_save_pred+"/IMAGES/IMAGE_num_"+str(i*batch_size+j)+"_truelabel_"+str(label[j].item())+".png")
                        for k in range(8):   
                            save_predictions(true_masks[j,k,...],output[j,k,...],probs[j,k,...],one_hot[j,k,...],pred_masks[j,k,...],k, max_label, compact, black_mask, None, i,j, batch_size,dir_save_pred, test=True)
        y_true_spec=[]
        y_pred_spec=[]
        
        for j in range(0,len(y_true), 15):
            pred_label=most_common(y_pred[j:j+15])
            true_label=y_true[j]
            y_true_spec.append(true_label)
            y_pred_spec.append(pred_label)
        dice_scores=np.asarray(dice_scores)
        seg_accuracies=np.asarray(seg_accuracies)
        n_specimens=np.asarray(n_specimens)

        #COMPUTE CONFUSION MATRIX
        confusion_mat=confusion_matrix(y_true_spec,y_pred_spec)

        #COMPUTE SEGMENTATION INDICES
        dice_scores=dice_scores/n_specimens
        seg_accuracies=seg_accuracies/n_specimens 
        
        
        #NORMALIZE CONFUSION MATRIX
        confusion_mat_norm=norm(confusion_mat)

        #COMPUTE MCA
        mca=mean_class_accuracy(confusion_mat) ##IS FOR UNET OUTPUTS
    
        print("CONFUSION MATRIX:")
        print(confusion_mat)
        print("CONFUSION MATRIX NORM:")
        print(confusion_mat_norm)
        print("MSA: ",seg_accuracies)
        print("MPA: ",mca)
        print("MDS: ", dice_scores)



def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--load', '-m', type=str, help='Load model from a .pth file')
    parser.add_argument('--load_data', type=int, default=1, help='Load dataloaders from a .pth file')
    parser.add_argument('--save_data', type=int, default=0, help='Save dataloaders into a .pth file')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=8, help='Number of classes')
    parser.add_argument('--channels', '-n', type=int, default=1, help='Number of input img channels')
    parser.add_argument('--fold', type=int, default=1, help='Number of experiment')
    parser.add_argument('--visualize', '-v', type=int, default=0, help='Visualize the outputs')
    parser.add_argument('--mathematical_morph', '-mm', type=int, default=0, help='Apply mathematical morphological operators on the output')




    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        test_net(net=net,
                  batch_size=args.batch_size,
                  device=device,
                  load_data=args.load_data,
                  save_data=args.save_data,
                  fold= args.fold,
                  visualize=args.visualize,
                  mathematical_morph=args.mathematical_morph
                  )
    except KeyboardInterrupt:
        raise
    