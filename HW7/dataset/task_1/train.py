import os
import torch
import torch.nn as nn
import numpy as np
import copy
from evaluator import evaluation_model
from util import get_test_conditions,save_image
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, D_model, G_model, z_dim, epochs, lr_d, lr_g):
    criterion_D = nn.BCELoss()
    criterion_G = nn.BCELoss()

    optimizer_D=torch.optim.Adam(D_model.parameters(),lr_d)
    optimizer_G=torch.optim.Adam(G_model.parameters(),lr_g)
    resnet_evaluation = evaluation_model()
    
    test_condition = get_test_conditions(os.path.join("test.json")).to(device)
    new_test_condition = get_test_conditions(os.path.join("new_test.json")).to(device)

    # print(len(test_condition))
    fixed_z = random_z(len(test_condition), z_dim).to(device)
    new_fixed_z = random_z(len(new_test_condition), z_dim).to(device)
    best_score = 0
    new_best_score = 0

    for epoch in range(1, epochs+1):
        total_loss_D = 0
        total_loss_G = 0
        for size, (images, condition) in enumerate(dataloader):
            D_model.train()
            G_model.train()
            batch_size = len(images)
            images = images.to(device, dtype=torch.float)
            # print(type(condition))
            condition = condition.to(device, dtype=torch.float)
            # print(type(condition))
            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            
            ##train discriminator
            optimizer_D.zero_grad()

            ##real images
            predict = D_model(images, condition)
            loss_real = criterion_D(predict, real)

            ##fake images
            latent_z = random_z(batch_size, z_dim).to(device)
            gen_imgs = G_model(latent_z, condition)
            predict = D_model(gen_imgs.detach(), condition)
            loss_fake = criterion_D(predict, fake)

            ##back propagation
            loss_D = loss_real+loss_fake
            # print(loss_D)
            loss_D.backward()
            optimizer_D.step()

            ##train generator
            for _ in range(4):
                optimizer_G.zero_grad()
                latent_z = random_z(batch_size, z_dim).to(device)
                gen_imgs = G_model(latent_z, condition)
                predict = D_model(gen_imgs, condition)
                loss_G = criterion_G(predict, real)
                ##bp
                loss_G.backward()
                optimizer_G.step()

            print(f'epoch{epoch} {size}/{len(dataloader)}  loss_G: {loss_G.item():.3f}  loss_D: {loss_D.item():.3f}')
            total_loss_G+=loss_G.item()
            total_loss_D+=loss_D.item()
        
        #evaluate test
        G_model.eval()
        D_model.eval()
        with torch.no_grad():
            gen_imgs=G_model(fixed_z,test_condition)
        score=resnet_evaluation.eval(gen_imgs,test_condition)
        if score>best_score:
            best_score=score
            best_model_wts=copy.deepcopy(G_model.state_dict())
            torch.save(best_model_wts,os.path.join('models/test',f'epoch{epoch}_score{score:.2f}.pt'))
                
        print(f'avg loss_g: {total_loss_G/len(dataloader):.3f}  avg_loss_d: {total_loss_D/len(dataloader):.3f}')
        print(f'testing score: {score:.2f}')
        print('---------------------------------------------')
        save_image(gen_imgs, os.path.join('gan_results/test', f'epoch{epoch}.png'), nrow=8, normalize=True)
       
        
        ## evaluate new test
        G_model.eval()
        D_model.eval()
        with torch.no_grad():
            new_gen_imgs=G_model(new_fixed_z, new_test_condition)
        new_score=resnet_evaluation.eval(new_gen_imgs, new_test_condition)
        if new_score>new_best_score:
            new_best_score=new_score
            new_best_model_wts=copy.deepcopy(G_model.state_dict())
            torch.save(new_best_model_wts,os.path.join('models/new_test',f'epoch{epoch}_score{new_score:.2f}.pt'))
        
        print(f'new testing score: {new_score:.2f}')
        print('---------------------------------------------')
        save_image(new_gen_imgs, os.path.join('gan_results/new_test', f'epoch{epoch}.png'), nrow=8, normalize=True)

def random_z(batch_size, z_dim):
    return torch.randn(batch_size,z_dim)