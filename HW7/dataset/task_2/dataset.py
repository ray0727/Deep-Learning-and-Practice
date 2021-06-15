import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def get_CelebA_data(root_folder):
    img_list = os.listdir(os.path.join(root_folder, 'CelebA-HQ-img'))
    label_list = []
    f = open(os.path.join(root_folder, 'CelebA-HQ-attribute-anno.txt'), 'r')
    num_imgs = int(f.readline()[:-1])
    attrs = f.readline()[:-1].split(' ')
    for idx in range(num_imgs):
        line = f.readline()[:-1].split(' ')
        label = line[2:]
        label = list(map(int, label))
        label_list.append(label)
    f.close()
    return img_list, label_list


class CelebALoader(data.Dataset):
    def __init__(self):
        self.root_folder = "/home/ray/Deep-Learning-and-Practice/HW7/dataset/task_2/"
        assert os.path.isdir(self.root_folder), '{} is not a valid directory'.format(self.root_folder)
        self.transformation = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        self.img_list, self.label_list = get_CelebA_data(self.root_folder)
        self.num_classes = 40
        print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        return len(self.label_list)


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_folder, 'CelebA-HQ-img', self.img_list[index])).convert("RGB")
        img = self.transformation(img)
        condition = self.label_list[index]
        condition = torch.tensor(condition)
        return img, condition


# a = CelebALoader()
# print((a[0][0]))
# print((a[0][1]))