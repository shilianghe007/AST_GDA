import torch
from .Conv_3 import Conv_3
from .resnet import ResNet18
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def create_model(data, n_class, arch, pretrained=False):
    """create classifier"""
    if arch == '4conv' or arch == '3conv': # 4+3
        model = Conv_3(data, n_class, arch)
    elif arch == 'resnet18':
        model = ResNet18(n_class)
        if pretrained:
            model_dict = model.state_dict()
            pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=True)
            # remove keys from pretrained dict that doesn't appear in model dict
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model.load_state_dict(pretrained_dict, strict=False)
    return model.to(device)