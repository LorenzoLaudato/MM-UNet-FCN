import numpy as np
import torch
from sklearn.utils import class_weight
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from sklearn.preprocessing import normalize
from dice_score import dice_coeff


def find_max_mask(pred_masks):
    max_label=0
    max_sum=0.0
    for i in range(7):
        sum=torch.sum(pred_masks[i,...]).item()
        if sum>max_sum:
            max_sum=sum
            max_label=i
    return max_label, max_sum
    
def mean_class_accuracy(confusion_matrix):
    count=confusion_matrix.sum(axis=1)
    true=np.diag(confusion_matrix)
    return np.mean(true/count)

    
def import_class_weights(train_labels,val_labels,test_labels):
    classes,_=np.unique(train_labels, return_counts=True)
    class_weights_train=class_weight.compute_class_weight('balanced',classes,train_labels)
    class_weights_train=torch.tensor(class_weights_train,dtype=torch.float32)
    class_weights_val=class_weight.compute_class_weight('balanced',classes,val_labels)
    class_weights_val=torch.tensor(class_weights_val,dtype=torch.float32)
    class_weights_test=class_weight.compute_class_weight('balanced',classes,test_labels)
    class_weights_test=torch.tensor(class_weights_test,dtype=torch.float32)
    return class_weights_train, class_weights_val, class_weights_test

def create_labels(train_loader, val_loader, test_loader, batch_size):
    train_labels=[]
    for item in train_loader:
        label = item['label']
        for i in range(batch_size):
            train_labels.append(label[i])
    train_labels=np.array(train_labels)
    val_labels=[]
    for item in val_loader:
        label = item['label']
        for i in range(batch_size):
            val_labels.append(label[i])
    val_labels=np.array(val_labels)
    test_labels=[]
    for item in test_loader:
        label = item['label']
        for i in range(batch_size):
            test_labels.append(label[i])
    test_labels=np.array(test_labels)
    return train_labels,val_labels,test_labels

def save_predictions(true_mask,output,probs,one_hot,pred_mask,k, max_label, compat, black_mask, epoch, i,j, batch_size, dir_save_pred, test=False):
    if not test:
        save_image(true_mask,dir_save_pred + "/TRUE_MASKS/TRUEMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        save_image(output.float(),dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK(OUTPUT)_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        save_image(probs.float(),dir_save_pred + "/PRED_MASKS/SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        save_image(one_hot.float(),dir_save_pred + "/PRED_MASKS/ONEHOT_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        save_image(pred_mask.float(),dir_save_pred + "/PRED_MASKS/OUTNET_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        if k==max_label:
            save_image(compat.float(),dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        elif k==7:
            save_image(one_hot.float(),dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        else:
            save_image(black_mask,dir_save_pred + "/PRED_MASKS/THRESH_SOFTMASK_epoch_"+str(epoch)+"_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
    else:
        save_image(true_mask,dir_save_pred + "/TRUE_MASKS/TRUEMASK_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        if k==max_label:
            save_image(compat.float(),dir_save_pred + "/PRED_MASKS/RESULT_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        elif k==7:
            save_image(one_hot.float(),dir_save_pred + "/PRED_MASKS/RESULT_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")
        else:
            save_image(black_mask,dir_save_pred + "/PRED_MASKS/RESULT_num_"+str(i*batch_size+j)+"_label_"+str(k)+".png")

def save_grids(n_masks):
    pred_masks=[]
    true_masks=[]
    for i in range(n_masks):
        for j in range(8):
            pred_mask=read_image("PRED_MASKS/RESULT_num_"+ str(i)+"_label_"+str(j)+".png")
            pred_masks.append(pred_mask)
            true_mask=read_image("TRUE_MASKS/TRUEMASK_num_"+ str(i)+"_label_"+str(j)+".png")
            true_masks.append(true_mask)
        total_list=true_masks+pred_masks
        grid=make_grid(total_list,nrow=8)
        save_image(grid.float(),"GRIDS/total_"+str(i)+".png")
        pred_masks=[]
        true_masks=[]
        
def norm(confusion_mat):
    return normalize(confusion_mat, axis=1,norm='l1').round(decimals=3)

def patch_reconstruction(masks):
    result = np.zeros((1040, 1388), np.uint8)
    for y in range(1388):
        for x in range(1040):
            if x < 328:
                if y < 251:
                    result[x][y] = masks[0][x][y]
                elif y < 384:
                    s = float(masks[0][x][y]) + float(masks[3][x][y - 251])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    result[x][y] = masks[3][x][y - 251]
                elif y < 635:
                    s = float(masks[6][x][y - 502]) + float(masks[3][x][y - 251])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    result[x][y] = masks[6][x][y - 502]
                elif y < 886:
                    s = float(masks[6][x][y - 502]) + float(masks[9][x][y - 753])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    result[x][y] = masks[9][x][y - 753]
                elif y < 1137:
                    s = float(masks[12][x][y - 1004]) + float(masks[9][x][y - 753])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    result[x][y] = masks[12][x][y - 1004]
                    
            elif x < 384:
                if y < 251:
                    s = float(masks[0][x][y]) + float(masks[1][x - 328][y])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 384:
                    s = float(masks[0][x][y]) + float(masks[3][x][y - 251]) + float(masks[1][x - 328][y]) + float(masks[4][x - 328][y - 251])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    s = float(masks[3][x][y - 251]) + float(masks[4][x - 328][y - 251])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 635:
                    s = float(masks[3][x][y - 251]) + float(masks[6][x][y - 502]) + float(masks[4][x - 328][y - 251]) + float(masks[7][x - 328][y - 502])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    s = float(masks[6][x][y - 502]) + float(masks[7][x - 328][y - 502]) 
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 886:
                    s = float(masks[6][x][y - 502]) + float(masks[9][x][y - 753]) + float(masks[7][x - 328][y - 502]) + float(masks[10][x - 328][y - 753])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    s = float(masks[9][x][y - 753]) + float(masks[10][x - 328][y - 753]) 
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 1137:
                    s = float(masks[9][x][y - 753]) + float(masks[12][x][y - 1004]) + float(masks[10][x - 328][y - 753]) + float(masks[13][x - 328][y - 1004])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    s = float(masks[12][x][y - 1004]) + float(masks[13][x - 328][y - 1004]) 
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                    
            elif x < 656:
                if y < 251:
                    result[x][y] = masks[1][x - 328][y]
                elif y < 384:
                    s = float(masks[1][x - 328][y]) + float(masks[4][x - 328][y - 251])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    result[x][y] = masks[4][x - 328][y - 251]
                elif y < 635:
                    s = float(masks[4][x - 328][y - 251]) + float(masks[7][x - 328][y - 502])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    result[x][y] = masks[7][x - 328][y - 502]
                elif y < 886:
                    s = float(masks[7][x - 328][y - 502]) + float(masks[10][x - 328][y - 753])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    result[x][y] = masks[10][x - 328][y - 753]
                elif y < 1137:
                    s = float(masks[13][x - 328][y - 1004]) + float(masks[10][x - 328][y - 753])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    result[x][y] = masks[13][x - 328][y - 1004]
                
            elif x < 712:
                if y < 251:
                    s = float(masks[1][x - 328][y]) + float(masks[2][x - 656][y]) 
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 384:
                    s = float(masks[1][x - 328][y]) + float(masks[4][x - 328][y - 251]) + float(masks[2][x - 656][y]) + float(masks[5][x - 656][y - 251])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    s = float(masks[4][x - 328][y - 251]) + float(masks[5][x - 656][y - 251])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 635:
                    s = float(masks[4][x - 328][y - 251]) + float(masks[7][x - 328][y - 502]) + float(masks[5][x - 656][y - 251]) + float(masks[8][x - 656][y - 502])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    s = float(masks[7][x - 328][y - 502]) + float(masks[8][x - 656][y - 502])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 886:
                    s = float(masks[7][x - 328][y - 502]) + float(masks[10][x - 328][y - 753]) + float(masks[8][x - 656][y - 502]) + float(masks[11][x - 656][y - 753])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    s = float(masks[10][x - 328][y - 753]) + float(masks[11][x - 656][y - 753])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 1137:
                    s = float(masks[10][x - 328][y - 753]) + float(masks[13][x - 328][y - 1004]) + float(masks[11][x - 656][y - 753]) + float(masks[14][x - 656][y - 1004])
                    if s >= 2:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    s = float(masks[13][x - 328][y - 1004]) + float(masks[14][x - 656][y - 1004])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                
            else: #y>=712
                if y < 251:
                    result[x][y] = masks[2][x - 656][y]
                elif y < 384:
                    s = float(masks[2][x - 656][y]) + float(masks[5][x - 656][y - 251])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 502:
                    result[x][y] = masks[5][x - 656][y - 251]
                elif y < 635:
                    s = float(masks[5][x - 656][y - 251]) + float(masks[8][x - 656][y - 502])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 753:
                    result[x][y] = masks[8][x - 656][y - 502]
                elif y < 886:
                    s = float(masks[8][x - 656][y - 502]) + float(masks[11][x - 656][y - 753])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                elif y < 1004:
                    result[x][y] = masks[11][x - 656][y - 753]
                elif y < 1137:
                    s = float(masks[14][x - 656][y - 1004]) + float(masks[11][x - 656][y - 753])
                    if s >= 1:
                        result[x][y] = 1
                    else: 
                        result[x][y] = 0
                else: #x>=1137
                    result[x][y] = masks[14][x - 656][y - 1004]
    return result

def most_common(lista):
    return max(set(lista), key=lista.count)

def update_seg_accuracies_and_dice_scores(dice_scores,seg_accuracies, n_specimens, spec_pred, spec_true,label, acc):
    seg_accuracies[label] += acc(spec_pred.int(), spec_true.int()).item()
    dice_scores[label] += dice_coeff(spec_pred, spec_true).item()
    n_specimens[label]+=1
    return dice_scores, seg_accuracies,n_specimens