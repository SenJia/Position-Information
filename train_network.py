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
from PIL import Image

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
from scipy.stats import spearmanr
from PIL import Image
from collections import defaultdict

from utils import loader
import models

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("train_dir", type=str, help='The datafolder of the training image, e.g., the DUT-S dataset.')
parser.add_argument("test_dir", type=str, help='The datafolder of the training image, e.g., the PASCAL-S dataset.')

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

parser.add_argument("--gpu", default="0", type=str)
parser.add_argument("--arch", default="vgg", type=str)
parser.add_argument("--pretrain", action="store_true", help='Initializing the backbone with an ImageNet pretrained one.')
parser.add_argument("--train_backbone", action="store_true", help='If the backbone is trainable.')
parser.add_argument("--save", action="store_true", help='If true, the model will be saved to the pre-defined folder.')

parser.add_argument("--feat_size", default=28, type=int, help='The features from the backbone will be re-scaled to a fixed size for alignment.')
parser.add_argument("--img_size", default=224, type=int, help='The input image and groundtruth will be re-scaled.')
parser.add_argument("--feat_index", default=[3, 8, 15, 22, 29], type=list, help='The index of the intermediate features, used for the VGG backbone.')
parser.add_argument("--vgg_pad", default=True, type=bool, help='If padding is applied for the VGG backbone.')
parser.add_argument("--decoder_pad", default=0, type=int, help='Default zero-padding used in the decoder, default is None.')
parser.add_argument("--decoder_depth", default=1, type=int, help='Default depth of the decoder.')
parser.add_argument("--prefix", default="", type=str, help='You might want to add some prefix to the name of output folder, e.g., depth_3 for depth equals 3.')

cudnn.benchmark = True
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


GT_PATTERNS = {
    "hor": "./synthetic/groundtruth/gt_hor.png",
    "ver": "./synthetic/groundtruth/gt_ver.png",
    "gau": "./synthetic/groundtruth/gt_gau.png",
    "horstp": "./synthetic/groundtruth/gt_horstp.png",
    "verstp": "./synthetic/groundtruth/gt_verstp.png",
    }

SYNTHETIC_IMG = {
    "black" : "./synthetic/images/black.jpg",
    "white" : "./synthetic/images/white.jpg",
    "noise" : "./synthetic/images/noise.jpg",
    }

def load_synthetic_imgs():
    ret = {}
    for name, path in SYNTHETIC_IMG.items():
        img = Image.open(path).convert('RGB')
        resized_img = F.resize(img, (args.img_size, args.img_size))
        # conter to tensor and expand the first dimension, batch.
        img_tensor = F.to_tensor(resized_img).unsqueeze(0)
        ret[name] = img_tensor
    return ret

def normalize(x):
    if x.min() != 0 or x.max() != 1:
        x -= x.min()
        x /= x.max()

def main(gt):

    def build_data_loader(data_root, gt_type, train=False):
        # load the synthetic groundtruth map.
        gt_path = GT_PATTERNS[gt_type]
        gt_pil = Image.open(gt_path).convert('L')
        gt_resized = F.resize(gt_pil, (args.img_size, args.img_size))
        gt_tensor = F.to_tensor(gt_resized)
     
        normalize(gt_tensor)

        #plt.imshow(gt_resized)
        #plt.show()
        #print (gt_tensor.min(), gt_tensor.max())

        dataset = loader.ImageList(data_root, transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
            ]))
        # create a dataloader for the images.
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=train,
            num_workers=args.workers, pin_memory=True)
        return data_loader, gt_tensor

    if args.arch.startswith("vgg"):
        print ("Building VGG and Decoder")
        backbone = models.vgg.vgg16(pretrained=args.pretrain, pad=args.vgg_pad, layer_index=args.feat_index) 
        decode_model = models.decoder.build_decoder(layer_list=[64, 128, 256, 512, 512], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)

    elif args.arch.startswith("res"):
        print ("Building {arch:} and Decoder".format(arch=args.arch))

        if "18" in args.arch:
            backbone = models.resnet.resnet18(pretrained=args.pretrain)
            decode_model = models.decoder.build_decoder(layer_list=[64, 128, 256, 512], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)
        elif "34" in args.arch:
            backbone = models.resnet.resnet34(pretrained=args.pretrain)
            decode_model = models.decoder.build_decoder(layer_list=[64, 128, 256, 512], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)
        elif "50" in args.arch:
            backbone = models.resnet.resnet50(pretrained=args.pretrain)
            decode_model = models.decoder.build_decoder(layer_list=[64*4, 128*4, 256*4, 512*4], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)
        elif "101" in args.arch:
            backbone = models.resnet.resnet101(pretrained=args.pretrain)
            decode_model = models.decoder.build_decoder(layer_list=[64*4, 128*4, 256*4, 512*4], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)
        elif "152" in args.arch:
            backbone = models.resnet.resnet152(pretrained=args.pretrain)
            decode_model = models.decoder.build_decoder(layer_list=[64*4, 128*4, 256*4, 512*4], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)

    elif args.arch.startswith("bag"):
        print ("Building BagNet and Decoder")
        backbone = models.bagnet.bagnet17(pretrained=args.pretrain)
        #backbone = models.bagnet.bagnet9(pretrained=args.pretrain)
        decode_model = models.decoder.build_decoder(layer_list=[64*4, 128*4, 256*4, 512*4], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)

    elif args.arch.startswith("res"):
        print ("Building Decoder only")
        backbone = None
        decode_model = models.decoder.build_decoder(layer_list=[3], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, decoder_depth=args.decoder_depth)

    trainable_params = []

    if not backbone is None:
        backbone = backbone.cuda()
        if not args.train_backbone:
            print ("Freezing the backbone.")
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()
        else:
            trainable_params.extend(list(backbone.parameters()))

    trainable_params.extend(list(decode_model.parameters()))

    decode_model = decode_model.cuda()

    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, trainable_params), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_loader, gt_map = build_data_loader(args.train_dir, gt_type=gt, train=True)
    test_loader, gt_map = build_data_loader(args.test_dir, gt_type=gt, train=False)

    # load the synthetic image, all white, all black...
    synthetic_img = load_synthetic_imgs()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, len(train_loader))
        train(train_loader, gt_map, backbone, decode_model, criterion, optimizer, epoch)

    # evalutea on the test set
    evaluate(test_loader, gt_map, backbone, decode_model, criterion, epoch)

    # evaluate on the synthetic images, white, black..
    evaluate_image(synthetic_img, gt_map, backbone, decode_model, criterion, epoch)

    if args.save:
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
    if args.train_backbone:
        backbone.train()
    decode_model.train()

    end = time.time()
    for i, (input) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # expand the gt_map to the batch size.
        batch_map = gt_map.expand(input.size(0), -1, -1, -1)
        #print (batch_map.shape, batch_map.max(), batch_map.min())

        # pass to gpu
        input = input.cuda()
        #print (input.shape)
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

def spearman(x, y):
    spc = []
    for pred, gt in zip(x, y):
        spc.append(spearmanr(pred.reshape(-1), gt.reshape(-1))[0])
    return spc 

def evaluate(test_loader, gt_map, backbone, decode_model, criterion, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    spc_list = []

    if args.train_backbone:
        backbone.eval()
    decode_model.eval()

    end = time.time()
    for i, (input) in enumerate(test_loader):
        data_time.update(time.time() - end)

        # expand the gt_map to the batch size.
        batch_map = gt_map.expand(input.size(0), -1, -1, -1)

        # pass to gpu
        input = input.cuda()

        if not backbone is None:
            input = backbone(input)

        output = decode_model(input).detach().cpu()

        mse = criterion(output, batch_map)
        losses.update(mse.item(), batch_map.size(0))

        spc_scores = spearman(output.squeeze().numpy(), batch_map.squeeze().numpy())
        spc_list.extend(spc_scores)


        batch_time.update(time.time() - end)
        end = time.time()

    print ("The MSE loss on the test set is", losses.avg)
    print ("The Spearman score on the test set is", np.mean(spc_list))

def evaluate_image(synthetic_dict, gt_map, backbone, decode_model, criterion, epoch):

    # expand the dimensionality of the gt_map.
    batch_map = gt_map.expand(1, -1, -1, -1)

    for name, img in synthetic_dict.items():
        if args.train_backbone:
            backbone.eval()
        decode_model.eval()

        # pass to gpu
        input = img.cuda()

        if not backbone is None:
            input = backbone(input)

        output = decode_model(input).detach().cpu()

        mse_score = criterion(output, batch_map)

        # the list has only one element, single image.
        spc_score = spearman(output.squeeze().numpy(), batch_map.squeeze().numpy())[0]

        print ("The MSE loss on", name, "is", mse_score.item())
        print ("The Spearman score on", name, "is", spc_score)

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
        print ("The target GT pattern is", gt)
        main(gt)
