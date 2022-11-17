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


def test_net(net1,net2,net3,
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
        net1.eval()
        net2.eval()
        net3.eval()

        ##INITIALIZE ARRAYS FOR PREDICTION
        y_true=[]
        y_pred1=[]
        y_pred2=[]
        y_pred3=[]

        # SUMS ARRAYS IS FOR MANAGE TIE BREAK, CHOOSE THE LABEL OF HIGHEST INTENSITY 
        y_sum1=[]
        y_sum2=[]
        y_sum3=[]
        specimen_true=[]
        specimen_pred1=[]
        specimen_pred2=[]
        specimen_pred3=[]
        n_specimens1=[0,0,0,0,0,0,0]
        seg_accuracies1=[0,0,0,0,0,0,0]
        dice_scores1=[0,0,0,0,0,0,0]
        n_specimens2=[0,0,0,0,0,0,0]
        seg_accuracies2=[0,0,0,0,0,0,0]
        dice_scores2=[0,0,0,0,0,0,0]
        n_specimens3=[0,0,0,0,0,0,0]
        seg_accuracies3=[0,0,0,0,0,0,0]
        dice_scores3=[0,0,0,0,0,0,0]
        n_specimens_total=[0,0,0,0,0,0,0]
        seg_accuracies_total=[0,0,0,0,0,0,0]
        dice_scores_total=[0,0,0,0,0,0,0]

        for i, item in enumerate(test_loader):
            img = item['image']/255
            true_masks=item['mask']
            label=item['label']
            ##TRANSFER TO GPU DEVICE 
            img = img.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                ##pred net1 
                pred_masks1,_ = net1(img)
                probs1 = F.softmax(pred_masks1, dim=1)
                one_hot1=F.one_hot(probs1.argmax(dim=1), net1.n_classes).permute(0,3,1,2)
                output1= (probs1>0.45).float()
                #pred net2
                pred_masks2,_ = net2(img)
                probs2 = F.softmax(pred_masks2, dim=1)
                one_hot2=F.one_hot(probs2.argmax(dim=1), net2.n_classes).permute(0,3,1,2)
                output2= (probs2>0.45).float()
                #pred net3
                pred_masks3,_ = net3(img)
                probs3 = F.softmax(pred_masks3, dim=1)
                one_hot3=F.one_hot(probs3.argmax(dim=1), net3.n_classes).permute(0,3,1,2)
                output3= (probs3>0.45).float()
                if mathematical_morph:
                    ## COMPUTE MATHEMATIC MORPHOLOGICAL OPERATIONS, YOU CAN SET DIM OF THE KERNEL SIZE
                    kernel=torch.ones(7,7)
                    one_hot1=opening(one_hot1.cpu(),kernel.cpu())
                    one_hot1=closing(one_hot1.cpu(),kernel.cpu())
                    one_hot2=opening(one_hot2.cpu(),kernel.cpu())
                    one_hot2=closing(one_hot2.cpu(),kernel.cpu())
                    one_hot3=opening(one_hot3.cpu(),kernel.cpu())
                    one_hot3=closing(one_hot3.cpu(),kernel.cpu())

                for j in range(batch_size):
                    max_label1, max_sum1=find_max_mask(probs1[j,...])
                    max_label2, max_sum2=find_max_mask(probs2[j,...])
                    max_label3, max_sum3=find_max_mask(probs3[j,...])
                    y_true.append(label[j].item())
                    y_pred1.append(max_label1)
                    y_pred2.append(max_label2)
                    y_pred3.append(max_label3)
                    y_sum1.append(max_sum1)
                    y_sum2.append(max_sum2)
                    y_sum3.append(max_sum3)
                    compact1=torch.zeros(384,384)
                    compact2=torch.zeros(384,384)
                    compact3=torch.zeros(384,384)
                    ###SUM ON 7 CHANNEL MASKS TO OBTAIN THE FINAL PREDICTED MASK
                    for l in range(7):
                        compact1+=one_hot1[j,l,:,:].cpu()
                        compact2+=one_hot2[j,l,:,:].cpu()
                        compact3+=one_hot3[j,l,:,:].cpu()
                    specimen_true.append(true_masks[j,label[j],:,:])
                    specimen_pred1.append(compact1)
                    specimen_pred2.append(compact2)
                    specimen_pred3.append(compact3)
                    ## EACH 15 PATCHES, WE RECONSTRUCT THE SPECIMEN
                    if (i*batch_size + j+1)%15==0:
                        spec_true=patch_reconstruction(specimen_true)
                        spec_pred1=patch_reconstruction(specimen_pred1)
                        spec_pred2=patch_reconstruction(specimen_pred2)
                        spec_pred3=patch_reconstruction(specimen_pred3)
                        spec_true=np.asarray(spec_true)
                        spec_pred1=np.asarray(spec_pred1)
                        spec_pred2=np.asarray(spec_pred2)
                        spec_pred3=np.asarray(spec_pred3)
                        spec_pred1=torch.as_tensor(spec_pred1, dtype=torch.float32)
                        spec_pred2=torch.as_tensor(spec_pred2, dtype=torch.float32)
                        spec_pred3=torch.as_tensor(spec_pred3, dtype=torch.float32)
                        spec_true=torch.as_tensor(spec_true, dtype=torch.float32)
                        specimen_true=[]
                        specimen_pred1=[]
                        specimen_pred2=[]
                        specimen_pred3=[]
                        if visualize==1:
                            save_image(spec_true,dir_save_pred+"/SPECIMENS/SPEC_TRUE_"+str(i*batch_size + j+1)+".png")
                            save_image(spec_pred1,dir_save_pred+"/SPECIMENS/SPEC_PRED1_"+str(i*batch_size + j+1)+".png")
                            save_image(spec_pred2,dir_save_pred+"/SPECIMENS/SPEC_PRED2_"+str(i*batch_size + j+1)+".png")
                            save_image(spec_pred3,dir_save_pred+"/SPECIMENS/SPEC_PRED3_"+str(i*batch_size + j+1)+".png")
                        spec_pred_total=torch.zeros(1040,1388)
                        #COMPUTE THE ENSEMBLING MASK
                        for h in range(spec_pred1.shape[0]):
                            for w in range(spec_pred1.shape[1]):
                                sum=spec_pred1[h,w]+spec_pred1[h,w]+spec_pred1[h,w]
                                if sum>1:
                                    spec_pred_total[h,w]=1
                                else:
                                    spec_pred_total[h,w]=0
                        dice_scores1,seg_accuracies1,n_specimens1=update_seg_accuracies_and_dice_scores(dice_scores1,seg_accuracies1,n_specimens1,spec_pred1,spec_true, label[j].item(), acc)
                        dice_scores2,seg_accuracies2,n_specimens2=update_seg_accuracies_and_dice_scores(dice_scores2,seg_accuracies2,n_specimens2,spec_pred2,spec_true, label[j].item(), acc)
                        dice_scores3,seg_accuracies3,n_specimens3=update_seg_accuracies_and_dice_scores(dice_scores3,seg_accuracies3,n_specimens3,spec_pred3,spec_true, label[j].item(), acc)
                        dice_scores_total,seg_accuracies_total,n_specimens_total=update_seg_accuracies_and_dice_scores(dice_scores_total,seg_accuracies_total,n_specimens_total,spec_pred_total,spec_true, label[j].item(), acc)


                    if visualize==1:
                        save_image(img[j,0,:,:].float(),dir_save_pred+"/IMAGES/IMAGE_num_"+str(i*batch_size+j)+"_truelabel_"+str(label[j].item())+".png")
                        for k in range(8):   
                            save_predictions(true_masks[j,k,...],output1[j,k,...],probs1[j,k,...],one_hot1[j,k,...],pred_masks1[j,k,...],k, max_label1, compact1, black_mask, None, i,j, batch_size,dir_save_pred, test=True)
                            save_predictions(true_masks[j,k,...],output2[j,k,...],probs2[j,k,...],one_hot2[j,k,...],pred_masks2[j,k,...],k, max_label2, compact2, black_mask, None, i,j, batch_size,dir_save_pred, test=True)
                            save_predictions(true_masks[j,k,...],output3[j,k,...],probs3[j,k,...],one_hot3[j,k,...],pred_masks3[j,k,...],k, max_label3, compact3, black_mask, None, i,j, batch_size,dir_save_pred, test=True)
        y_true_spec=[]
        y_pred_spec=[]

        for j in range(0,len(y_true), 15):
            pred_label1=most_common(y_pred1[j:j+15])
            index1=y_pred1[j:j+15].index(pred_label1)
            pred_label2=most_common(y_pred2[j:j+15])
            index2=y_pred2[j:j+15].index(pred_label2)
            pred_label3=most_common(y_pred3[j:j+15])
            index3=y_pred3[j:j+15].index(pred_label3)
            pred_labels=[pred_label1, pred_label2,pred_label3]
            if pred_labels.count(max(set(pred_labels), key=pred_labels.count))>1:
                pred_label=most_common(pred_labels)
            else:
                ### MANAGE TIE BREAK
                sums=[y_sum1[j+index1], y_sum2[j+index2], y_sum3[j+index3]]
                index=sums.index(max(sums))
                pred_label=pred_labels[index]
            true_label=y_true[j]
            y_true_spec.append(true_label)
            y_pred_spec.append(pred_label)
        print(y_pred_spec)

        dice_scores1=np.asarray(dice_scores1)
        dice_scores2=np.asarray(dice_scores2)
        dice_scores3=np.asarray(dice_scores3)
        dice_scores_total=np.asarray(dice_scores_total)
        seg_accuracies1=np.asarray(seg_accuracies1)
        seg_accuracies2=np.asarray(seg_accuracies2)
        seg_accuracies3=np.asarray(seg_accuracies3)
        seg_accuracies_total=np.asarray(seg_accuracies_total)
        n_specimens1=np.asarray(n_specimens1)
        n_specimens2=np.asarray(n_specimens2)
        n_specimens3=np.asarray(n_specimens3)
        n_specimens_total=np.asarray(n_specimens_total)

        #COMPUTE SEGMENTATION INDICES
        dice_scores1=dice_scores1/n_specimens1
        dice_scores2=dice_scores2/n_specimens2
        dice_scores3=dice_scores3/n_specimens3
        dice_scores_total=dice_scores_total/n_specimens_total
        seg_accuracies1=seg_accuracies1/n_specimens1 
        seg_accuracies2=seg_accuracies2/n_specimens2
        seg_accuracies3=seg_accuracies3/n_specimens3 
        seg_accuracies_total=seg_accuracies_total/n_specimens_total 
        
        
        #COMPUTE CONFUSION MATRIX
        confusion_mat=confusion_matrix(y_true_spec,y_pred_spec)
        confusion_mat_norm=norm(confusion_mat)

        #COMPUTE MCA
        mca=mean_class_accuracy(confusion_mat)
    
        print("CONFUSION MATRIX:")
        print(confusion_mat)
        print("CONFUSION MATRIX NORM:")
        print(confusion_mat_norm)
        print("MSA net1: ",seg_accuracies1)
        print("MSA net2: ",seg_accuracies2)
        print("MSA net3: ",seg_accuracies3)
        print("MSA total: ",seg_accuracies_total) ##EVALUATION ON ENSEMBLING MASK
        print("MPA: ",mca)
        print("MDS net1: ", dice_scores1)
        print("MDS net2: ", dice_scores2)
        print("MDS net3: ", dice_scores3)
        print("MDS total: ", dice_scores_total) ##EVALUATION ON ENSEMBLING MASK



def get_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--load1', type=str, help='Load model1 from a .pth file')
    parser.add_argument('--load2', type=str, help='Load model2 from a .pth file')
    parser.add_argument('--load3', type=str, help='Load model3 from a .pth file')
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
    net1 = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    net2 = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    net3 = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    

    logging.info(f'Network:\n'
                 f'\t{net1.n_channels} input channels\n'
                 f'\t{net1.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net1.bilinear else "Transposed conv"} upscaling')

    if args.load1 and args.load2 and args.load3:
        net1.load_state_dict(torch.load(args.load1, map_location=device))
        logging.info(f'Model1 loaded from {args.load1}')
        net2.load_state_dict(torch.load(args.load2, map_location=device))
        logging.info(f'Model2 loaded from {args.load2}')
        net3.load_state_dict(torch.load(args.load3, map_location=device))
        logging.info(f'Model3 loaded from {args.load3}')

    net1.to(device=device)
    net2.to(device=device)
    net3.to(device=device)
    try:
        test_net(net1=net1,net2=net2, net3=net3,
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
    