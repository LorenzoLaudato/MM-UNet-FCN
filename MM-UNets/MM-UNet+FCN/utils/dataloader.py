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
from torchvision.utils import save_image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()  # interactive mode


class HEP2Dataset(Dataset):
    """HEp-2 dataset."""

    def __init__(self, csv_file, root_dir, transform=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): whether apply or not DA.
        """
        old_frame = pd.read_csv(csv_file, names=["Image", "Mask", "Label", "Intensity"])
        self.total_old = len(old_frame)
        self.frame = pd.DataFrame(columns=["Image", "Mask", "Label", "Intensity","Aug"])
        if transform:
            for i in range(self.total_old):
                #ADD A ROW FOR EACH TECHNIQUE YOU WANT TO APPLY 
                row_0={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 0}
                row_1={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 1}
                self.frame = self.frame.append(row_0, ignore_index=True)
                self.frame = self.frame.append(row_1, ignore_index=True)
                if old_frame.loc[i]["Label"] in ["numem ","golgi ","mitsp "]:
                    row_2={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 2}
                    row_3={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 3}
                    row_4={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 4}
                    self.frame = self.frame.append(row_2, ignore_index=True)
                    self.frame = self.frame.append(row_3, ignore_index=True)
                    self.frame = self.frame.append(row_4, ignore_index=True)
                    if old_frame.loc[i]["Label"] in ["mitsp ","golgi "]:
                        row_5={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 5}
                        row_6={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 6}
                        self.frame = self.frame.append(row_5, ignore_index=True)
                        self.frame = self.frame.append(row_6, ignore_index=True)
                        if old_frame.loc[i]["Label"] in ["golgi "]:
                            row_7={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 7}
                            self.frame = self.frame.append(row_7, ignore_index=True)
                            row_8={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 8}
                            self.frame = self.frame.append(row_8, ignore_index=True)
                            row_9={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 9}
                            self.frame = self.frame.append(row_9, ignore_index=True)


        else:
            for i in range(self.total_old):
            #for i in range(20):
                row_0={'Image':old_frame.loc[i]["Image"],'Mask':old_frame.loc[i]["Mask"],'Label':old_frame.loc[i]["Label"],'Intensity': old_frame.loc[i]["Intensity"],'Aug': 0}
                self.frame = self.frame.append(row_0, ignore_index=True)
        self.total = len(self.frame)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        aug = os.path.join(self.root_dir, str(self.frame.iloc[idx, 4]))
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])
        image = io.imread(img_name)
        mask_name = os.path.join(self.root_dir, self.frame.iloc[idx, 1])
        mask = io.imread(mask_name)

        #IMPLEMENT AUGMENTATIONS TECHNIQUE
        if aug == 1:
            image = imutils.rotate(image, 90)
            mask = imutils.rotate(mask, 90)
        elif aug == 2:
            image = imutils.rotate(image, 180)
            mask = imutils.rotate(mask, 180)
        #aug == 3:
        elif aug==3:
            image = imutils.rotate(image, 270)
            mask = imutils.rotate(mask, 270)
        elif aug==4:
            image = cv2.flip(image,0)
            mask = cv2.flip(mask,0)
        elif aug==5:
            image = cv2.flip(image,-1)
            mask = cv2.flip(mask,-1)
        elif aug==6:
            image = cv2.flip(image,1)
            mask = cv2.flip(mask,1)
        elif aug==7:
            image = imutils.rotate(image, 90)
            mask = imutils.rotate(mask, 90)
            image = cv2.flip(image,0)
            mask = cv2.flip(mask,0)
        elif aug==8:
            image = imutils.rotate(image, 270)
            mask = imutils.rotate(mask, 270)
            image = cv2.flip(image,0)
            mask = cv2.flip(mask,0)
        elif aug==9:
            image = imutils.rotate(image, 90)
            mask = imutils.rotate(mask, 90)
            image = cv2.flip(image,1)
            mask = cv2.flip(mask,1)
        
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
        intensity = self.frame.iloc[idx, 3]
        
        if intensity == 'positive':
            inten = 0
        else:
            inten = 1
               
        image = image[newaxis, :, :]
        image = torch.as_tensor(image, dtype=torch.float32)
        lab = torch.as_tensor(int(lab), dtype=torch.int16)
        inten = torch.as_tensor(int(inten), dtype=torch.int16)

        return {
            'image': image,
            'mask': masks,
            'label':lab
        }