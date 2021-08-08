# This is the model definition for the simple readout, 
# we call it PosENet when directly training on raw images.
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo
import math
import torch

class Decoder(nn.Module):
    def __init__(self, layer_list, size_mid=None, size_out=None, padding=0, filter_size=3, decoder_depth=1):
        """
        Arguments:
        layer_list: a list of the multi-level feature, e.g., [64, 128, 256]. The features will be resized and concatenated into one tensor
                    for the convolution module.
        size_mid: the size of the concatenated multi-level feature.
        size_out: the size of the output map.
        padding: the size of zero-padding area, should be 0 as used in our paper.
        filter_size: the size of the conv filter used in the decoder module.
        deocder_depth: the number of conv layers used in the decoder module.
        """
        super(Decoder, self).__init__()

        self.size_mid = size_mid
        self.size_out = size_out


        self.num_filters = sum(layer_list) # the multi-level feature will be concatenated into one tensor, on which the conv module will be applied.

        print ("Padding in the decoder", padding, "Kernel size", filter_size, "Number of filters in the Decoder", self.num_filters)

        self.output = self._make_output(self.num_filters, filter_size, padding, decoder_depth)

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

    def _make_output(self, num_filters, size, padding, decoder_depth):
        layers = []
        for i in range(decoder_depth):
            if i == decoder_depth - 1:
                layers += nn.Conv2d(num_filters, 1, kernel_size=size, padding=padding),
                layers += nn.BatchNorm2d(1),
                layers.append(nn.Sigmoid())
            else:
                layers += nn.Conv2d(num_filters, num_filters, kernel_size=size, padding=padding),
                layers += nn.BatchNorm2d(num_filters),
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # num_filters equals three when the input is an RGB image.
        if self.num_filters == 3: 
            feat = F.interpolate(x, self.size_mid, mode='bilinear', align_corners=True)
        # otherwise, the input is a list of multi-level features.
        else:
            for i in range(len(x)):
                x[i] = F.interpolate(x[i], self.size_mid, mode='bilinear', align_corners=True)
            feat = torch.cat(x, dim=1)

        feat = self.output(feat) 
        feat = F.interpolate(feat, self.size_out, mode='bilinear', align_corners=True)
        return feat 

def build_decoder(model_path=None, **kargs):
    decoder = Decoder(**kargs)
    if model_path:
        state = torch.load(model_path)["state_dict"]
        decoder.load_state_dict(state)

    return decoder
