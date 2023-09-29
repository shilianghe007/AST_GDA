
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
import copy

device = torch.device(
            f"cuda:0" if torch.cuda.is_available() else "cpu"
        )

class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args):
        super(ResNet, self).__init__(*args)
        self._out_features = self.fc.in_features

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = x.view(-1, self._out_features)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def _resnet(arch, block, layers, pretrained, progress):
    model = ResNet(block, layers)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress)



class FC(nn.Module):
    def __init__(self, in_shape, hdim=512, n_class=10, reg=True):
        super(FC, self).__init__()

        self.reg = reg
        self.in_shape = in_shape

        self.fc1 = nn.Linear(in_shape, hdim)
        self.fc4 = nn.Linear(hdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, n_class)
        self.batch_norm1 = nn.BatchNorm1d(hdim)
        self.batch_norm2 = nn.BatchNorm1d(n_class)
        
        if self.reg:
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout2 = nn.Dropout(p=0.25)
        
    def forward(self, x):
        x = x.reshape(-1, self.in_shape)
        x = F.relu(self.batch_norm1(self.fc1(x)))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc2(x))
        if self.reg:
            x = self.dropout1(x)
            x = self.dropout2(self.batch_norm2(self.fc3(x)))
        else:
            x = self.fc3(x)

        return x


class SIMPLE_CNN(nn.Module):
    def __init__(self, mode, reg=True):
        super(SIMPLE_CNN, self).__init__()

        self.reg = reg
        if mode == "mnist":
            self.conv1 = nn.Conv2d(1, 32, 3)
            self.fc1 = nn.Linear(288, 128)
        elif mode == "cifar":
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.fc1 = nn.Linear(512, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.fc2 = nn.Linear(128, 10)
        self.flatten = nn.Flatten()
        if self.reg:
            self.dropout1 = nn.Dropout(p=0.5)
            self.dropout2 = nn.Dropout(p=0.25)
            self.batch_norm = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        if self.reg:
            x = self.dropout1(x)
            x = self.dropout2(self.batch_norm(self.fc2(x)))
        else:
            x = self.fc2(x)

        return x

class ENCODER(nn.Module):
    def __init__(self, layers=4, resnet=False):
        super(ENCODER, self).__init__()

        if resnet:
            self.encode = nn.Sequential(
                nn.Conv2d(1, 3, 1),
                nn.ReLU(),
                resnet18(pretrained=True),
                nn.Flatten()
            )
        else:
            kernel_size = 5 if layers == 3 else 3
            stride = 2 if layers == 3 else 1
            padding = 2 if layers == 3 else 1
            self.encode = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                )
            for i in range(layers-1):
                self.encode.add_module(
                    'conv_{0}'.format(i+1),
                    nn.Sequential(
                        nn.Conv2d(32, 32, kernel_size, stride=stride, padding=padding),
                        nn.ReLU()
                    )
                )
            self.encode.add_module(
                'dropout',
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.BatchNorm2d(32),
                    nn.Flatten(),
                )
            )

    def forward(self, x):
        x = self.encode(x)
        return x
    
    
class MLP(nn.Module):
    def __init__(self, mode, n_class, indim, hidden=1024, layers=1):
        super(MLP, self).__init__()

        dim=indim
        if layers == 1:
            self.mlp = nn.Linear(dim, n_class)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
            )
            for i in range(layers-2):
                self.mlp.add_module(
                    'linear_{0}'.format(i+1),
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                    )
                )
            self.mlp.add_module('linear_{0}'.format(layers-1), nn.Linear(hidden, n_class))
        
    def forward(self, x):
        return self.mlp(x)


class Classifier(nn.Module):
    def __init__(self, encoder, mlp):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.mlp = mlp
        
    def forward(self, x, return_inter=False):
        o_encoder = self.encoder(x)
        if return_inter:
            return self.mlp(o_encoder), o_encoder
        else:
            return self.mlp(o_encoder)


def Conv_3(data, n_class, arch):
    """create classifier"""
    if arch == '4conv': # 4+3
        encoder = ENCODER(layers=4)
        indim = 25088 if data=='mnist' else 32768
        model = Classifier(encoder, MLP(mode=data, n_class=n_class, indim=indim, hidden=1024, layers=3))
    elif arch == '3conv': # 3+1
        encoder = ENCODER(layers=3)
        indim = 512
        model = Classifier(encoder, MLP(mode=data, n_class=n_class, indim=indim, hidden=1024, layers=1))
    return model
