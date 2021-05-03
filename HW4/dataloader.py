import pandas as pd
from torchvision import transforms,models
from torch.utils import data
from PIL import Image
import numpy as np
import os

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

# transforms.Resize((256, 256))
class RetinopathyLoader(data.Dataset):
    def __init__(self, img_path, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_path = img_path
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transformation = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.ToTensor(), 
                                      transforms.Normalize((0.37, 0.26, 0.18),(0.25, 0.17, 0.12))])
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        single_img_name = os.path.join(self.img_path, self.img_name[index]+ '.jpeg')
        single_img =  Image.open(single_img_name)
        img = self.transformation(single_img)
        label = self.label[index]
        return img, label
