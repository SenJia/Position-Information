import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


__all__ = [
    'VGG', 'vgg16',
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG(nn.Module):

    def __init__(self, features=None, layer_index=None):
        super(VGG, self).__init__()

        self.features = features

        if layer_index:
            self.layer_index = layer_index

    def forward(self, x):
        output = []
        for i, l in enumerate(self.feat):
            x = l(x)
            if i in self.layer_index:
                output.append(x)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False, pad=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if pad:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def vgg16(pretrain=True, pad=True, **kwargs):

    model = VGG(make_layers(cfg['D'], pad=pad), **kwargs)

    if pretrain:
        # take the pre-trained backbone
        model_state = model.state_dict()
        loaded = model_zoo.load_url(model_urls['vgg16'])
        pretrained = {k:v for k,v in loaded.items() if k in model_state}
        model_state.update(pretrained)
        model.load_state_dict(model_state)
        print ("Updated parameters", len(pretrained)) 

    for i, (n, v) in enumerate(model.named_parameters()):
        print (i, n)
    print ("Chekcing the index for the multi-level features.")

    feat_lst = []
    for k, v in model.features.named_children():
        feat_lst.append(v)
    model.feat = nn.ModuleList(feat_lst)
    del model.features
    return model

