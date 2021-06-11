import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label

class ICLEVRLoader(data.Dataset):
    def __init__(self, mode='train'):
        self.root_folder = "/home/ray/Deep-Learning-and-Practice/HW7/dataset/task_1/"
        self.mode = mode
        self.transformation = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.img_list, self.label_list = get_iCLEVR_data(self.root_folder,mode)
        
        print("> Found %d images..." % (len(self.label_list)))
        self.num_classes = 24
                
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label_list)

    def __getitem__(self, index):
        if self.mode == "train":
            img = Image.open(os.path.join(self.root_folder, 'images', self.img_list[index])).convert("RGB")
            img = self.transformation(img)
            condition = self.label_list[index]
            return img, condition
        else:
            condition = self.label_list[index]
            return condition

# a = ICLEVRLoader()
# print(a[3])
