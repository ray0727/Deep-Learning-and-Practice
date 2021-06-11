import os
import torch
from torch.utils.data import DataLoader

from dataset import ICLEVRLoader
from model import Generator,Discriminator
from train import train

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim=100
c_dim=100
image_shape=(3,64,64)
epochs=200
lr=0.001
batch_size=64

if __name__=='__main__':

    # load training data
    dataset_train=ICLEVRLoader(mode="train")
    loader_train=DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=2)

    # create generate & discriminator
    generator=Generator(z_dim,c_dim).to(device)
    discrimiator=Discriminator(image_shape,c_dim).to(device)
    generator.weight_init(mean=0,std=0.02)
    discrimiator.weight_init(mean=0,std=0.02)

    # train
    train(loader_train,discrimiator,generator,z_dim,epochs,lr)