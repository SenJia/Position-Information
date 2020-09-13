# Data loader for our position information project.
# This data loader is expecting a text file wehre each line is
# a relative path of the image.
# A root path is needed which will be concatenated with the image path.
# Author : Sen Jia 
#

import torch.utils.data as data

from PIL import Image
from PIL import ImageFilter
import os
import os.path
import numpy as np
import torch
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from torch import nn

def make_dataset(root, txt_file):
    images = []
    with open(txt_file,"r") as f:
        for line in f:
            path = line.rstrip("\n")
            if " " in path:
                path = path.split(" ")[0]
            images.append(root / path)
    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageList(data.Dataset):
    def __init__(self, root, txt_file, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(root, txt_file)
        if not imgs:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        img_path = self.imgs[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img


    def __len__(self):
        return len(self.imgs)

