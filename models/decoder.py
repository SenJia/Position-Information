# This is the model definition for the simple readout, 
# we call it PosENet when directly training on raw images.
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import math
import torch

class Decoder(nn.Module):
    def __init__(self, layers, size_mid=None, size_out=None, readout=1, padding=0, stride=1, size=3, depth=1):
        super(Decoder, self).__init__()
        self.readout = readout

        self.size_mid = size_mid
        self.size_out = size_out
        self.padding = padding
        self.stride = stride
        self.size = size

        self.num_layers = sum(layers)

        print ("Padding in the decoder", self.padding, "Kernel size", self.size, "Number of filters in the Decoder", self.num_layers)

        self.output = self._make_output(self.num_layers, size=self.size, depth=depth)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_output(self, planes, size, readout=1, depth=1):
        layers = []
        for i in range(depth):
            if i == depth - 1:
                layers += nn.Conv2d(planes, readout, kernel_size=size, stride=self.stride, padding=self.padding),
                layers += nn.BatchNorm2d(readout),
                layers.append(nn.Sigmoid())
            else:
                layers += nn.Conv2d(planes, planes, kernel_size=size, stride=self.stride, padding=self.padding),
                layers += nn.BatchNorm2d(planes),
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.num_layers == 3: 
            feat = F.interpolate(x, self.size_mid, mode='bilinear')
            feat = self.output(feat)
        elif self.multi < 0:
            for i in range(len(x)):
                x[i] = F.interpolate(x[i], self.size_mid, mode='bilinear')
            feat = torch.cat(x, dim=1)
            feat = self.output(feat) 
        else:
            feat = F.interpolate(x[self.multi], self.size_mid, mode='bilinear')
            feat = self.output(feat)
        feat = F.interpolate(feat, self.size_out, mode='bilinear')
        return feat 

def build_decoder(model_path=None, **kargs):
    decoder = Decoder(**kargs)
    if model_path:
        state = torch.load(model_path)["state_dict"]
        decoder.load_state_dict(state)

    return decoder
