from __future__ import print_function, division
import os
import cv2
import torch
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import imutils
from numpy import newaxis
import numpy as np
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()  # interactive mode


class HEP2Dataset(Dataset):
    """HEp-2 dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        cuts={'homogeneous ':9, 'speckled ': 9, 'nucleolar ': 9, 'centromere ': 9, 'golgi ': 49, 'numem ': 24, 'mitsp ': 34}
        old_frame = pd.read_csv(csv_file, names=["Image","Mask","Label"])
        self.total_old = len(old_frame)
        self.frame = pd.DataFrame(columns=["Image","Mask","Label"])
        for i in range(self.total_old):
        #for i in range(20):
            row_0={'Image':old_frame.loc[i]["Image"],"Mask":old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"]}
            self.frame = self.frame.append(row_0, ignore_index=True)
            for j in range(cuts[old_frame.loc[i]["Label"]]):
                row={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"]}
                self.frame = self.frame.append(row, ignore_index=True)
        self.total = len(self.frame)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        #print('__getitem__')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        specimen = io.imread(img_name)
        
        mask_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        mask_specimen = io.imread(mask_name)

        #RANDOM POINT 
        x1= random.randint(192, specimen.shape[1]-192)
        y1= random.randint(192, specimen.shape[0]-192)
        shape= (specimen.shape[1], specimen.shape[0])
        
        #BUILD ROTATION MATRIX 
        matrix=cv2.getRotationMatrix2D(center=(x1,y1), angle=random.randint(0,360), scale=1)
        
        #EXTRACT IMAGE E MASK ROTATED
        image=cv2.warpAffine(src=specimen, M=matrix, dsize=shape)
        mask=cv2.warpAffine(src=mask_specimen, M=matrix, dsize=shape)
        x=int(x1-192)
        y=int(y1-192)
        
        #EXTRACT IMAGE E MASK PATCH
        image=image[y:y+384, x:x+384]
        mask=mask[y:y+384, x:x+384]
        label = self.frame.iloc[idx, 2]
        masks=[]
        
        #CREATE BLACK MASK E PUT IN ARRAY OF MASKS 
        black_mask=np.zeros(mask.shape)

        for i in range(7):
            masks.append(black_mask)
        mask[mask > 1.0] = 1.0 #If pixel value is greater than 1, pixelvalue=1 
        
        #BUILD BACKGROUND MASK
        temp=np.ones(mask.shape)
        bg_mask= temp-mask

        #PUT IN ARRAY MASKS THE GROUNDTRUTH MASK IN THE INDEX OF THE LABEL TRUE
        if label == 'homogeneous ':
            masks[0]= mask
            lab = 0
        elif label == 'speckled ':
            masks[1]= mask
            lab = 1
        elif label == 'nucleolar ':
            masks[2]= mask
            lab = 2
        elif label == 'centromere ':
            masks[3]= mask
            lab = 3
        elif label == 'golgi ':
            masks[4]= mask
            lab = 4
        elif label == 'numem ':
            masks[5]= mask
            lab = 5
        else: # mistp
            masks[6]= mask
            lab = 6
            
        #ADD IN THE LAST ELEMENT OF ARRAY THE BACKGROUND MASK
        masks.append(bg_mask)
        masks=np.asarray(masks)
        masks=torch.as_tensor(masks, dtype=torch.float32)
          
        image = image[newaxis, :, :]
        image = torch.as_tensor(image, dtype=torch.float32)
        lab = torch.as_tensor(int(lab), dtype=torch.int16)

        return {
            'image': image,
            'mask': masks,
            'label':lab
        }