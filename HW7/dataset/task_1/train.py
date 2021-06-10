import os
import torch
import torch.nn as nn
import numpy as np
import copy
from evaluator import evaluation_model
from util import get_test_conditions,save_image
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, D_model, G_model, z_dim, epochs, lr):
    criterion_D = nn.BCELoss()
    criterion_G = nn.BCELoss()

    optimizer_D=torch.optim.Adam(d_model.parameters(),lr)
    optimizer_G=torch.optim.Adam(g_model.parameters(),lr)
    evaluation = evaluation_model()
    test_condition = get_test_conditions(os.path.join("test.json")).to(device)

    for epoch in range(1, epochs+1):
        total_loss_D = 0
        total_loss_G = 0
        for _, (images, condition) in enumerate(dataloader):
            D_model.train()
            G_model.train()
            batch_size = len(images)
            images = images.to(device, dtype=torch.float)
            condition = condition.to(device, dtype=torch.long)
            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            
            ##train discriminator
            optimizer_D_.zero_grad()

            ##real images
            predict = D_model(images)
            loss_real = criterion_D(predict, real)

            ##fake images
            latent_z = random_z(batch_size, z_dim).to(device)
            
            loss_fake = criterion_D(predict, fake)
        

def random_z(batch_size, z_dim):
    return torch.randn(batch_size,z_dim)