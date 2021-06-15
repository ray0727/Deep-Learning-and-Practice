import os
import torch
from torch.utils.data import DataLoader
import argparse
from dataset import ICLEVRLoader
from model import Generator,Discriminator
from train import train

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim=100
c_dim=200
image_shape=(3,64,64)
epochs=200
lr_g=0.00005
lr_d=0.0002
batch_size=32

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test_mode', default="test", type=str)
    args = parser.parse_args()
    #print(args.test_mode)
    # load training data
    dataset_train=ICLEVRLoader(mode="train")
    loader_train=DataLoader(dataset_train,batch_size=batch_size,shuffle=True,num_workers=2)

    # create generate & discriminator
    generator=Generator(z_dim,c_dim).to(device)
    discrimiator=Discriminator(image_shape,c_dim).to(device)
    generator.weight_init(mean=0,std=0.02)
    discrimiator.weight_init(mean=0,std=0.02)

    mode = args.test_mode
    # train
    train(loader_train,discrimiator,generator,z_dim,epochs,lr_d,lr_g, mode)