"""
Author  : Xu fuyong
Time    : created by 2019/7/16 20:17

"""
import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import SRCNN
from datasets import DRealSRDataset, TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

'''
python train2.py --train-file 

'''
LR_path = r"E:\sensingX\super-resolution\DRealSR\x2\Train_x2\train_LR"
HR_path = r"E:\sensingX\super-resolution\DRealSR\x2\Train_x2\train_HR"


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    return args

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr*0.1}
    ], lr=args.lr)

    dataset = DRealSRDataset(LR_path, HR_path)
    exit()

    # train_dataset = TrainDataset(args.train_file)
    # train_dataloader = DataLoader(dataset=train_dataset,
    #                               batch_size=args.batch_size,
    #                               shuffle=True,
    #                               num_workers=args.num_workers,
    #                               pin_memory=True,
    #                               drop_last=True)

    # eval_dataset = EvalDataset(args.eval_file)
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))

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

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))