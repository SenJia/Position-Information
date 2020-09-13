# This file is used to evaluate the pre-trained decoder(simple readout) on natural images o
# or other types of synthetic images.(check folder ./synthetic/images)
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
from scipy.stats import spearmanr
from PIL import Image

from utils import loader
import models

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')

parser.add_argument("model_path", type=pl.Path)
parser.add_argument("--arch", default="vgg", type=str)
#parser.add_argument("--gpu", default="1", type=str)

parser.add_argument("--feat_size", default=28, type=int, help='The features from the backbone will be re-scaled to a fixed size for alignment.')
parser.add_argument("--img_size", default=224, type=int, help='The input image and groundtruth will be re-scaled.')
parser.add_argument("--feat_index", default=[3, 8, 15, 22, 29], type=list, help='The index of the intermediate features, used for the VGG backbone.')
parser.add_argument("--vgg_pad", default=True, type=bool, help='If padding is applied for the VGG backbone.')
parser.add_argument("--decoder_pad", default=0, type=int, help='Default zero-padding used in the decoder, default is None.')
parser.add_argument("--decoder_depth", default=1, type=int, help='Default depth of the decoder.')

cudnn.benchmark = True
args = parser.parse_args()

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

DATA_DIR = # assign the data folder here.

GT_PATTERNS = {
    "hor": "./synthetic/groundtruth/gt_hor.png",
    "ver": "./synthetic/groundtruth/gt_ver.png",
    "gau": "./synthetic/groundtruth/gt_gau.png",
    "horstp": "./synthetic/groundtruth/gt_horstp.png",
    "verstp": "./synthetic/groundtruth/gt_verstp.png",
    }

"""
# you might want to change this
VAL_DATA = {
            "dataset name":"path of your txt file, please see the readme file.",
           }
"""

SYN_DATA = {"BLACK": "./synthetic/images/black.jpg",
            "WHITE": "./synthetic/images/white.jpg",
            "NOISE": "./synthetic/images/noise.jpg",
           }

def main():
    def load_gt(gt_type):
        # load the synthetic groundtruth map.
        gt_path = GT_PATTERNS[gt_type]
        gt_pil = Image.open(gt_path).convert('L')
        gt_resized = F.resize(gt_pil, (args.img_size, args.img_size))
        gt_tensor = F.to_tensor(gt_resized)
        if gt_tensor.min() != 0 or gt_tensor.max() != 1:
            gt_tensor -= gt_tensor.min()
            gt_tensor /= gt_tensor.max()
        return gt_tensor

    def build_data_loader(data_root, data_file, train=False):
        # create a dataloader for the images.
        data_loader = torch.utils.data.DataLoader(
            loader.ImageList(data_root, data_file, transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
            ]),
            ),
            batch_size=args.batch_size, shuffle=train,
            num_workers=args.workers, pin_memory=True)
        return data_loader

    # the last flag in the folder name is the groundtruth type. 
    gt = str(args.model_path).split("_")[-1]
    model_path = args.model_path / "decoder.pth.tar"
    if args.arch.startswith("vgg"):
        print ("Building VGG and Decoder")

        backbone = models.vgg.vgg16(pad=args.vgg_pad, layer_index=args.feat_index) 

        decode_model = models.decoder.build_decoder(model_path=model_path, layers=[64,128,256,512,512], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, depth=args.decoder_depth)

    elif args.arch.startswith("res"):
        print ("Building ResNet and Decoder")
        backbone = models.resnet.resnet152()
        decode_model = models.decoder.build_decoder(model_path=model_path, layers=[64*4, 128*4, 256*4, 512*4], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, depth=args.decoder_depth)
    elif args.arch.startswith("img"):
        print ("Building Decoder only")
        backbone = None
        decode_model = models.decoder.build_decoder(model_path=model_path, layers=[3], size_mid=(args.feat_size, args.feat_size), size_out=(args.img_size, args.img_size), padding=args.decoder_pad, depth=args.decoder_depth)

    if not backbone is None:
        for param in backbone.parameters():
            param.requires_grad = False

        backbone = backbone.cuda()
        backbone.eval()
    for param in decode_model.parameters():
        param.requires_grad = False
    decode_model = decode_model.cuda()
    decode_model.eval()

    gt_tensor = load_gt(gt)

    # specify your own natural image dataset in VAL_DATA.
    """
    for k, val_file in VAL_DATA.items():
        val_loader = build_data_loader(DATA_DIR, val_file, gt) 
        print ("Validating on", k)
        validate(val_loader, backbone, decode_model, gt_tensor)
    """

    for k, path  in SYN_DATA.items():
        print ("Validating on", k)
        validate_img(path, backbone, decode_model, gt_tensor)

def mae(pred, gt):
    return np.mean(np.abs(pred-gt))

def spc(pred, gt):
    return spearmanr(pred.flatten(), gt.flatten())[0]

def validate(val_loader, backbone, decode_model, gt_map):

    end = time.time()

    gt_map = gt_map.squeeze().numpy()

    mae_lst = []
    spc_lst = []

    for i, (input) in enumerate(val_loader):

        # pass to gpu
        input = input.cuda()

        if not backbone is None:
            input = backbone(input)

        output = decode_model(input)
        for pred in output:
            pred = pred.squeeze().cpu().numpy()
            m  = mae(pred, gt_map)
            s  = spc(pred, gt_map)
            mae_lst.append(m)
            spc_lst.append(s)
            #plt.imshow(pred)
            #plt.show()
    print ("Average MAE:", np.mean(mae_lst), "Average SPC", np.mean(spc_lst))

def validate_img(path, backbone, decode_model, gt_map):
    img_pil = Image.open(path).convert('RGB')
    img_resized = F.resize(img_pil, (args.img_size, args.img_size))
    img_tensor = F.to_tensor(img_resized)

    # expand the dimension
    input = img_tensor.cuda().unsqueeze(0)
    gt_map = gt_map.squeeze().numpy()

    if not backbone is None:
        input = backbone(input)
    pred = decode_model(input).squeeze().cpu().numpy()
    m  = mae(pred, gt_map)
    s  = spc(pred, gt_map)

    print ("MAE:", m, "SPC", s)

if __name__ == '__main__':

    main()
