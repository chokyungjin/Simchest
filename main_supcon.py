from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
from datasets.datasets_Supcon import ClassPairDataset
import random
import numpy as np
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=11,
                        help='number of training epochs')
                        
    parser.add_argument('--train_path', type=str, 
    default="/mnt/nas125_cxr_fu/cleansing_datasets/baseline_followup_pair_4class/")

    parser.add_argument('--fov', type=str, default=None)
    parser.add_argument('--aug', type=str, default=None)
    parser.add_argument('--margin', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='real')
    parser.add_argument('--name', type=str, default='real')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                            help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='5,15,25',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', type=str, default=None)

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    # opt.data_folder = './datasets/'
    opt.model_path = './save/{}/{}_models'.format(opt.name, opt.dataset)
    opt.tb_path = './save/{}/{}_tensorboard'.format(opt.name, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 3
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    print('[*] option setting is finished')
    return opt

def set_FU_loader(opt):
    if opt.fov == 'True':
        train_datasets = ClassPairDataset(opt.train_path, dataset=opt.dataset, fov=opt.fov, margin=opt.margin,  aug=opt.aug, mode='train')
    else:
        train_datasets = ClassPairDataset(opt.train_path, dataset=opt.dataset, fov=None, margin=None,  aug=None, mode='train')
    
    print('[*] train data property')
    train_datasets.get_data_property()
    
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=opt.batch_size, 
    num_workers=opt.num_workers, pin_memory=True, shuffle=True)

    
    return train_loader

def set_model(opt):
    model = SupConResNet(name=opt.model) 
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    idx = 0
    end = time.time()
    for base_img_1, pair_img_1, change_labels, labels, _ in iter(train_loader):
        idx +=1
        data_time.update(time.time() - end)
        images = torch.cat([base_img_1, pair_img_1], dim=0)
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        # features = features.view(bsz, 2, -1)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method == 'SupCon':
            loss = criterion(features, labels)
            # loss = criterion(features, labels, change_labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))
    
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

def main():
    opt = parse_option()

    if opt.random_seed is not None:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        torch.backends.cudnn.deterministic = True

    # build data loader
    # train_loader = set_loader(opt)
    train_loader = set_FU_loader(opt)
    print('[*] train_loader setting is finished')

    # build model and criterion
    model, criterion = set_model(opt)
    print('[*] model setting is finished')
    
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.resume is True:
        checkpoint = torch.load(opt.pretrained)
        start_epochs = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        print('[*] model load is finished')
    else:
        start_epochs = 1
    # training routine
    for epoch in range(start_epochs, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()