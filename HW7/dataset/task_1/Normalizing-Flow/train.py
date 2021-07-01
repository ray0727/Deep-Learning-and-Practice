import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from dataset import ICLEVRLoader
import util
import copy

from test_util import get_test_conditions, save_image
from evaluator import evaluation_model

import wandb
from models import Glow
from tqdm import tqdm


def main(args):
    # Set up main device and scale batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    wandb.init(project="NF_task1")


    trainset = ICLEVRLoader(mode="train")
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # Model
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps,)
    net = net.to(device)
    
    wandb.watch(net)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net, args.gpu_ids)
    #     cudnn.benchmark = args.benchmark

    start_epoch = 1
    # if args.resume:
    #     # Load checkpoint.
    #     print('Resuming from checkpoint at ckpts/best.pth.tar...')
    #     assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('ckpts/best.pth.tar')
    #     net.load_state_dict(checkpoint['net'])
    #     global best_loss
    #     global global_step
    #     best_loss = checkpoint['test_loss']
    #     start_epoch = checkpoint['epoch']
    #     global_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))

    train(args.num_epochs, net, trainloader, device, optimizer, scheduler, loss_fn, args.max_grad_norm)

@torch.enable_grad()
def train(epochs, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    
    net.train()
    loss_meter = util.AverageMeter()
    test_condition=get_test_conditions(os.path.join('test.json')).to(device)
    new_test_condition=get_test_conditions(os.path.join('new_test.json')).to(device)

    # for i in range(len(test_condition)):
    #     print(test_condition[i])

    resnet_evaluation = evaluation_model()
    best_score = 0
    new_best_score = 0
    for epoch in range(1, epochs+1):
        print('\nEpoch: %d' % epoch)
        with tqdm(total=len(trainloader.dataset)) as progress_bar:
            for x, cond_x in trainloader:
                net.train()
                x , cond_x= x.to(device, dtype=torch.float), cond_x.to(device, dtype=torch.float)
                optimizer.zero_grad()
                
                z, sldj = net(x, cond_x, reverse=False)
                loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                loss.backward()
                wandb.log({"loss meter": loss_meter.avg})
                if max_grad_norm > 0:
                    util.clip_grad_norm(optimizer, max_grad_norm)
                optimizer.step()
                scheduler.step(global_step)

                progress_bar.set_postfix(nll=loss_meter.avg,
                                        bpd=util.bits_per_dim(x, loss_meter.avg),
                                        lr=optimizer.param_groups[0]['lr'])
                progress_bar.update(x.size(0))
                # print(x.size(0))
                global_step += x.size(0)
                # print(global_step)
        ##evaluate
        net.eval()
        with torch.no_grad():
            gen_imgs = sample(net, test_condition, device)

        score=resnet_evaluation.eval(gen_imgs, test_condition)
        if score > best_score:
            best_score = score
            best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(best_model_wts,os.path.join('weight/test',f'epoch{epoch}_score{score:.2f}.pt'))


        net.eval()
        with torch.no_grad():
            new_gen_imgs = sample(net, new_test_condition, device)

        new_score=resnet_evaluation.eval(new_gen_imgs, new_test_condition)
        if new_score > new_best_score:
            new_best_score = new_score
            new_best_model_wts = copy.deepcopy(net.state_dict())
            torch.save(new_best_model_wts,os.path.join('weight/new_test',f'epoch{epoch}_score{new_score:.2f}.pt'))

        print(f'testing score: {score:.3f}')
        print(f'new_testing score: {new_score:.3f}')
        wandb.log({"score": score})
        wandb.log({"new_score": new_score})
        save_image(gen_imgs, os.path.join('images/test', f'epoch{epoch}.png'), nrow=8, normalize=True)
        save_image(new_gen_imgs, os.path.join('images/new_test', f'epoch{epoch}.png'), nrow=8, normalize=True)


@torch.no_grad()
def sample(net, condition, device):
    B = len(condition)
    # print(B)
    z = torch.randn((B, 3, 64, 64), dtype=torch.float, device=device)
    net.eval()
    x, _ = net(z, condition, reverse=True)
    x = torch.sigmoid(x)
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on task1')

    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=512, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=4, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=6, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=500, type=int, help='Number of epochs to train')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    global_step = 0
    main(parser.parse_args())
