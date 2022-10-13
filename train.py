"""
Author  : Xu fuyong
Time    : created by 2019/7/16 20:17

"""
import argparse
import os
import copy
# import time

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import SRCNN
from datasets import DRealSRDataset, TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

import wandb

dataset_dir = "./datasets"
train_name = "x2_train_4"   # hdf5文件输入路径
val_name = "x2_val_1"       # hdf5文件输入路径
outputs_dir = f"./outputs"  # 模型输出路径
run_name = "x2_train_4"     # wandb run名称


scale = 2
lr = 1e-4
batch_size = 16
num_workers = 0
num_epochs = 100
seed = 42

train_file = os.path.join(dataset_dir, train_name)
val_file = os.path.join(dataset_dir, val_name)
outputs_dir = os.path.join(outputs_dir, f'x{scale}')



if __name__ == '__main__':

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr*0.1}
    ], lr=lr)


    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    val_dataset = EvalDataset(val_file)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    global_step = 0

    # init wandb    
    experiment = wandb.init(project='SUPER-RESOLUTION-SRCNN', resume='allow', anonymous='never', name = run_name)
    experiment.config.update(dict(epochs=num_epochs, 
                                  batch_size=batch_size, 
                                  learning_rate=lr,                            
                                  scale=scale,
                                  num_workers=num_workers,
                                  seed=seed                 
                                  ))
    
    print(f'''---Starting training---
        Epochs:          {num_epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_dataset)}
        Validation size: {len(val_dataset)}
        Device:          {device.type}
        Scale:           {scale}
        Workers number:  {num_workers}
        Seed:            {seed}
    ''')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, num_epochs - 1))
            
            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                global_step += 1
                
                experiment.log({
                    'train loss' : loss.item(),
                    'step' : global_step,
                    'epoch' : epoch
                })

        torch.save(model.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in val_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation psnr': epoch_psnr.avg,
            'LR': wandb.Image(inputs[0].cpu()),
            'HR': {
                'true': wandb.Image(labels[0].float().cpu()),
                'pred': wandb.Image(preds[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch,
            **histograms
        })
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))