# The training code for our ICLR position project.
# It will run on all the five groundtruth types, but you can change it only for one type, e.g., hor.
# The backbone used, VGG or ResNet, is pre-trained on ImageNet by default.
# Our work is to if the position information is encoded in the pre-trained model.
# A simple read-out is trainable, which will be saved in the specified folder, please pay attention 
# to the argument prefix, you might want to specify some symbols for clarity.
# 

import argparse
import os
import shutil
import time
import pathlib as pl

import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
from PIL import Image
from collections import defaultdict

from utils import loader
import models

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=300, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr-decay', default=[10], type=list,
                    metavar='N', help='print frequency (default: 5)')

parser.add_argument("--arch", default="vgg", type=str)
#parser.add_argument("--gpu", default="1", type=str)

parser.add_argument("--feat_size", default=28, type=int, help='The features from the backbone will be re-scaled to a fixed size for alignment.')
parser.add_argument("--img_size", default=224, type=int, help='The input image and groundtruth will be re-scaled.')
parser.add_argument("--feat_index", default=[3, 8, 15, 22, 29], type=list, help='The index of the intermediate features, used for the VGG backbone.')
parser.add_argument("--vgg_pad", default=True, type=bool, help='If padding is applied for the VGG backbone.')
parser.add_argument("--decoder_pad", default=0, type=int, help='Default zero-padding used in the decoder, default is None.')
parser.add_argument("--decoder_depth", default=1, type=int, help='Default depth of the decoder.')
parser.add_argument("--prefix", default="", type=str, help='You might want to add some prefix to the name of output folder, e.g., depth_3 for depth equals 3.')

cudnn.benchmark = True
args = parser.parse_args()

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

DATA_DIR = # pass the folder of your data here.

GT_PATTERNS = {
    "hor": "./synthetic/groundtruth/gt_hor.png",
    "ver": "./synthetic/groundtruth/gt_ver.png",
    "gau": "./synthetic/groundtruth/gt_gau.png",
    "horstp": "./synthetic/groundtruth/gt_horstp.png",
    "verstp": "./synthetic/groundtruth/gt_verstp.png",
    }


def main(gt):

    def build_data_loader(data_root, data_file, gt_type, train=False):
        # load the synthetic groundtruth map.
        gt_path = GT_PATTERNS[gt_type]
        gt_pil = Image.open(gt_path).convert('L')
        gt_resized = F.resize(gt_pil, (args.img_size, args.img_size))
        gt_tensor = F.to_tensor(gt_resized)
        if gt_tensor.min() != 0 or gt_tensor.max() != 1:
            gt_tensor -= gt_tensor.min()
            gt_tensor /= gt_tensor.max()
 
        # create a dataloader for the images.
        data_loader = torch.utils.data.DataLoader(
            loader.ImageList(data_root, data_file, transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
            ]),
            ),
            batch_size=args.batch_size, shuffle=train,
            num_workers=args.workers, pin_memory=True)
        return data_loader, gt_tensor

    if args.arch.startswith("vgg"):
        print ("Building VGG and Decoder")

        backbone = models.vgg.vgg16(pad=args.vgg_pad, layer_index=args.feat_index) 

        decode_model = models.decoder.build_decoder(layers=[64,128,256,512,512], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, depth=args.decoder_depth)

    elif args.arch.startswith("res"):
        print ("Building ResNet and Decoder")
        backbone = models.resnet.resnet152()
        decode_model = models.decoder.build_decoder(layers=[64*4, 128*4, 256*4, 512*4], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, depth=args.decoder_depth)
    elif args.arch.startswith("img"):
        print ("Building Decoder only")
        backbone = None
        decode_model = models.decoder.build_decoder(layers=[3], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, depth=args.decoder_depth)

    if not backbone is None:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone = backbone.cuda()
        backbone.eval()

    decode_model = decode_model.cuda()

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, decode_model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_file = # pass the text file here, each line should contain one image path. 
    train_loader, gt_map = build_data_loader(DATA_DIR, train_file, gt_type=gt, train=True)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, len(train_loader))
        train(train_loader, gt_map, backbone, decode_model, criterion, optimizer, epoch)


    output_folder = str(args.arch)+"_"+gt
    if args.prefix:
        output_folder = args.prefix + "_" + output_folder
    print ("Output directory", output_folder)
    if output_folder and not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    state = {
        'state_dict' : decode_model.state_dict(),
        }
    path = os.path.join(output_folder, "decoder.pth.tar")
    save_model(state, path)

def save_model(state, path):
    torch.save(state, path)

def train(train_loader, gt_map, backbone, decode_model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, (input) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # expand the gt_map to the batch size.
        batch_map = gt_map.expand(input.size(0), -1, -1, -1)

        # pass to gpu
        input = input.cuda()
        batch_map = batch_map.cuda()

        if not backbone is None:
            input = backbone(input)

        output = decode_model(input)

        mse = criterion(output, batch_map)

        losses.update(mse.item(), batch_map.size(0))

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, len_epoch):
    if epoch in args.lr_decay:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

if __name__ == '__main__':

    GTs = ["hor", "ver", "gau", "horstp", "verstp"] 
    for gt in GTs:
        main(gt)
