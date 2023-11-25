from torchvision import models
import torch.nn as nn
import sys
import torch
import numpy as np
import sys
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os

sys.path.append("./utils/")
from custom_clip import load_clip_state_dict, ResNet_LPIPS 
from CBR import CBR_LargeT

from slim_resnet import slim_resnet50
from slim_densenet import slim_densenet201

SUPPORTED_WEIGHTS = ["imagenet", "CheXpert", "random", "bimcv"]

def truncate_densenet(net: nn.Module, trunc: int):
    if trunc==-1:
        return net
    else:
        # print(list(net..children()))
        # asdfg
        return nn.Sequential(*list(net.children())[:4+trunc*2])
    

def densenet121(pretrained : str ="imagenet", trunc : int=-1, classes : int=7):
    # sainity check
    assert pretrained in SUPPORTED_WEIGHTS, "weights not found"
    
    if pretrained=="imagenet":
        print("load imagenet pretrained weights")
        net = models.densenet121(pretrained=True).features
    elif pretrained=="CheXpert":
        print("load chexpert pretrained weights")
        net = torch.load("../checkpoints/CheXpert/densenet121_random_-1_1/best.pt").module.to(torch.device("cpu"))
    elif pretrained=="random":
        net = models.densenet121(pretrained=False).features
    else:
        sys.exit("wrong specification")
        
    net = truncate_densenet(net, trunc)
    test_input = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(net(test_input).shape)
    fc = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                          out_features=classes,
                          bias=True)
            ])
    net = nn.Sequential(net, fc)
        
    return net


def densenet201(pretrained : str ="imagenet", trunc : int=-1, classes : int=7):
    # sainity check
    assert pretrained in SUPPORTED_WEIGHTS, "weights not found"
    
    if pretrained=="imagenet":
        print("load imagenet pretrained weights")
        net = models.densenet201(pretrained=True).features
    elif pretrained=="CheXpert":
        print("load chexpert pretrained weights")
        net = torch.load("../checkpoints/CheXpert/densenet201_random_-1_1/best.pt").module.to(torch.device("cpu"))
    elif pretrained=="random":
        net = models.densenet201(pretrained=False).features
    else:
        sys.exit("wrong specification")
        
    net = truncate_densenet(net, trunc)
    test_input = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(net(test_input).shape)
    fc = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                          out_features=classes,
                          bias=True)
            ])
    net = nn.Sequential(net, fc)

    return net




def truncate_resnet(net: nn.Module, trunc: int, imagenet=False):
    if trunc==-1:
        return net
    else:
        # print(list(net..children()))
        # asdfg
        return nn.Sequential(*list(net.children())[:4+trunc])




def unsqueeze_net(net, return_net=False, sequential_break=True):
    ret_list = []
    # print(net[0])
    for x in net:
        if ((isinstance(x, nn.Conv2d) or isinstance(x, nn.BatchNorm2d) \
            or isinstance(x, nn.MaxPool2d) or isinstance(x, nn.AdaptiveAvgPool2d) \
            or isinstance(x, nn.ReLU) or isinstance(x, nn.Linear)
            or isinstance(x, models.resnet.Bottleneck))) \
            or ((not sequential_break) and isinstance(x, nn.Sequential))\
            or isinstance(x, timm.models.efficientnet_blocks.InvertedResidual)\
            or isinstance(x, timm.models.efficientnet_blocks.InvertedResidual):
            ret_list.append(x)
        else:
            ret_list.extend(list(x.children()))
    if return_net:
        ret_list = nn.Sequential(*ret_list)
    return ret_list

def find_next_trunc(net, idx):
    l = list(net.children())
    for i, layer in enumerate(l[idx:]):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, models.resnet.Bottleneck):
            idx += i
            break
    return nn.Sequential(*list(net.children())[:idx]), idx

def layer_truncate_resnet(net: nn.Module, imagenet=False):
    net = [net]
    net = unsqueeze_net(net, sequential_break=True)
    net = unsqueeze_net(net, sequential_break=True)
    pre_count = 0
    while len(net) != pre_count:
        pre_count = len(net)
        print(len(net))
        net = unsqueeze_net(net)
    net = nn.Sequential(*net)
    p = 1
    idx = []
    while p <= len(net):
        _, next_p = find_next_trunc(net, p)
        p = next_p + 1
        idx.append(next_p-1)

    return net, idx





def resnet50(pretrained : str ="imagenet", trunc : int=-1, classes : int=7, args=None, layer_wise=False, freeze=False):
    # sainity check
    assert pretrained in SUPPORTED_WEIGHTS, "weights not found"
    
    if pretrained=="imagenet":
        net = models.resnet50(pretrained=True)
        net.fc = nn.Identity()
    elif pretrained=="CheXpert":
        print("load chexpert pretrained weights")
        net = torch.load("../checkpoints/CheXpert/resnet50_random_-1_low_1/best.pt").module.to(torch.device("cpu"))
    elif pretrained=="random":
        net = models.resnet50(pretrained=False)
        net.fc = nn.Identity()
    elif pretrained == "bimcv":
        net = torch.load("../checkpoints/BIMCV/resnet50_imagenet_-1_1/best.pt").module.to(torch.device("cpu"))
        net = list(net.children())[0]
        # net = nn.Sequential(*list(net.children())[0])
    else:
        sys.exit("wrong specification")
    
    if not layer_wise:
        net = truncate_resnet(net, trunc)
    else:
        net, idx = layer_truncate_resnet(net)
        true_trunc = idx[trunc-1]
        net = nn.Sequential(*list(net.children())[:true_trunc+1])

    test_input = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(net(test_input).shape)
    if freeze:
        for name, param in net.named_parameters():
            param.requires_grad = False
    
    if args != None and args.pooling:
        linear_input_dim = np.prod(net(test_input).shape[:2])
        print("pooling!!", linear_input_dim)
        linear = nn.Linear(in_features=linear_input_dim,
                    out_features=classes,
                    bias=True)
        torch.nn.init.kaiming_normal_(linear.weight) 
        fc = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            linear,
        ])
    else:
        linear = nn.Linear(in_features=linear_input_dim,
                    out_features=classes,
                    bias=True)
        torch.nn.init.kaiming_normal_(linear.weight) 
        fc = nn.Sequential(nn.Flatten(),
                    linear)

    
    net = nn.Sequential(net, fc)
    return net


def truncate_clipresnet(net: nn.Module, trunc: int):
    if trunc==-1:
        return net
    else:
        net = nn.Sequential(*list(net.children())[:11+trunc])
        return net

def clip_resnet50(trunc: int=-1, classes : int=7):
    vision_layers = (3, 4, 6, 3)
    vision_width = 64
    image_resolution = 224
    embed_dim = 1024
    vision_heads = vision_width * 32 // 64

    net = ResNet_LPIPS(layers=vision_layers,
                         output_dim=embed_dim,
                         heads=vision_heads,
                         input_resolution=image_resolution,
                         width=vision_width).to(torch.device("cuda"))

    preprocess = load_clip_state_dict(net, clip_model_name="RN50")
    net = net.to(torch.device("cpu"))
    net = truncate_clipresnet(net, trunc)

    test_input = test_input = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(net(test_input).shape)
    
    
    fc = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                        out_features=classes,
                        bias=True)
            ])
    net = nn.Sequential(net, fc)
    return net, preprocess


## big transfer

def truncate_bitresnet(net: nn.Module, trunc: int):
    if trunc==-1:
        return net
    else:
        net = nn.Sequential(
            *list(net.children())[0],
            *list(net.stages.children())[:trunc])
        return net

def bit_resnet50(trunc: int=-1, classes : int=7):
    # net = timm.create_model('resnetv2_50x1_bitm', pretrained=True)
    net = timm.create_model('resnetv2_50x3_bitm_in21k', pretrained=True)
    
    net.eval() ## used in the timm document
    net = truncate_bitresnet(net, trunc)

    test_input = test_inplut = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(net(test_input).shape)
    
    
    fc = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                        out_features=classes,
                        bias=True)
            ])
    net = nn.Sequential(net, fc)

    config = resolve_data_config({}, model=net)
    transform = create_transform(**config)

    return net, transform


def freeze_resnet50(finetune_from: int=1, classes: int=7):
    net = resnet50(pretrained="imagenet", trunc=-1, classes=classes)
    for name, param in net.named_parameters():
        print(name)
        if finetune_from > 0:
            if name == "0.bn1.weight" or name == "0.conv1.weight" or name == "0.bn1.bias":
                param.requires_grad = False
                print(name, "freeze")
            for i in range(1, finetune_from):
                if "layer" + str(i) in name:
                    param.requires_grad = False
                    print(name, "freeze")
        else:
            break
    return net

def layer_wise_freeze_resnet50(pretrained : str ="imagenet", finetune_from : int=-1, classes : int=7, args=None, layer_wise=False):
    # sainity check
    assert pretrained in SUPPORTED_WEIGHTS, "weights not found"
    
    if pretrained=="imagenet":
        net = models.resnet50(pretrained=True)
        net.fc = nn.Identity()
    elif pretrained=="CheXpert":
        print("load chexpert pretrained weights")
        net = torch.load("../checkpoints/CheXpert/resnet50_random_-1_low_1/best.pt").module.to(torch.device("cpu"))
    elif pretrained=="random":
        net = models.resnet50(pretrained=False)
        net.fc = nn.Identity()
    elif pretrained == "bimcv":
        net = torch.load("../checkpoints/BIMCV/resnet50_imagenet_-1_1/best.pt").module.to(torch.device("cpu"))
        net = list(net.children())[0]
        # net = nn.Sequential(*list(net.children())[0])
    else:
        sys.exit("wrong specification")
    
    net, idx = layer_truncate_resnet(net)

    print(idx, len(idx))
    true_finetune_from = idx[finetune_from-1]
    freezed = nn.Sequential(*list(net.children())[:true_finetune_from+1])
    for param in freezed.parameters():
        param.requires_grad = False

    unfreezed = nn.Sequential(*list(net.children())[true_finetune_from+1:])

    fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048,
                        out_features=classes,
                        bias=True)
        )
    return nn.Sequential(freezed, unfreezed, fc)


def freeze_densenet201(finetune_from: int=1, classes: int=7):
    net = densenet201(pretrained="imagenet", trunc=-1, classes=classes)
    for name, param in net.named_parameters():
        print(name)
        if finetune_from > 0:
            if name == "0.norm0.weight" or name == "0.conv0.weight" or name == "0.norm0.bias":
                param.requires_grad = False
                print(name, "freeze")
            for i in range(1, finetune_from):
                if "block" + str(i) in name or "transition" + str(i) in name:
                    param.requires_grad = False
                    print(name, "freeze")
        else:
            break
    return net


def cbr_larget(pretrained : str ="imagenet", classes: int=7):
    assert pretrained == "imagenet", "not support other pretrained weights"
    pretrained_weights = torch.load("/home/le/TL/checkpoints/ImageNet/CBR_LargeT_V2.pt")
    net = CBR_LargeT(input_channels=3, output_class=1000)
    net = nn.DataParallel(net)
    net.load_state_dict(pretrained_weights)
    net = net.module.to(torch.device("cpu"))
    in_channels = net.fc.in_features
    net.fc = nn.Linear(in_channels, classes, bias=True)

    return net


def count_parameters(model, require_grad=True):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if require_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def truncate_alexnet(net, trunc):
    if trunc == -1:
        return net
    else:
        tmp_dict = [3, 6, 8, 10, 13, 2, 5, 7]
        if trunc <=5:
            print(trunc, tmp_dict[trunc])
            return nn.Sequential(*list(net.features.children())[:tmp_dict[trunc-1]])
        else:
            return nn.Sequential(
                list(net.children())[0],
                list(net.children())[1],
                list(net.children())[2][:tmp_dict[trunc-1]])


def alexnet(pretrained : str ="imagenet", trunc : int=-1, classes : int=7):
    # sainity check
    assert pretrained in SUPPORTED_WEIGHTS, "weights not found"
    
    if pretrained=="imagenet":
        print("load imagenet pretrained weights")
        net = models.alexnet(pretrained=True)
    elif pretrained=="CheXpert":
        print("load chexpert pretrained weights")
        net = torch.load("../checkpoints/CheXpert/densenet121_random_-1_1/best.pt").module.to(torch.device("cpu"))
    elif pretrained=="random":
        net = models.alexnet(pretrained=False)
    else:
        sys.exit("wrong specification")
        
    net = truncate_alexnet(net, trunc)
    test_input = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    print(net(test_input).shape)
    linear_input_dim = np.prod(net(test_input).shape)
    if trunc <=5 and trunc > 0:
        pool_dim = int(np.sqrt(linear_input_dim / 9216))
        linear_input_dim = np.prod(nn.MaxPool2d(kernel_size=pool_dim, stride=pool_dim)(net(test_input)).shape)
        fc = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=pool_dim, stride=pool_dim),
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                          out_features=classes,
                          bias=True)
            ])
    else:
        fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                          out_features=classes,
                          bias=True)
            )
    net = nn.Sequential(net, fc)
        
    return net

def find_next_trunc_efficientnet(net, idx):
    l = list(net.children())
    for i, layer in enumerate(l[idx:]):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, timm.models.efficientnet_blocks.InvertedResidual)\
            or isinstance(layer, timm.models.efficientnet_blocks.InvertedResidual):
            idx += i
            # print(type(layer))
            break
    return nn.Sequential(*list(net.children())[:idx]), idx

def layer_truncate_efficient(net: nn.Module, imagenet=False, unsqueeze_num=3):
    net = [net]
    for i in range(unsqueeze_num):
        net = unsqueeze_net(net, sequential_break=True)
    net = nn.Sequential(*net)
    p = 1
    idx = []
    while p <= len(net):
        _, next_p = find_next_trunc_efficientnet(net, p)
        p = next_p + 1
        idx.append(next_p-1)

    return net, idx


def truncate_efficientnet(net, trunc=-1):
    net, idx = layer_truncate_efficient(net)
    net = nn.Sequential(*net)
    if trunc == -1:
        return net
    else:
        # print(len(list(net.children())))
        # asdf
        return nn.Sequential(*list(net.children())[:2+trunc])

def efficientnet(trunc, num_classes=7, pretrained='imagenet'):
    assert pretrained in SUPPORTED_WEIGHTS, "weights not found"
    
    if pretrained=="imagenet":
        print("load imagenet pretrained weights")
        net = timm.create_model('efficientnet_b0', pretrained=True)
    elif pretrained=="random":
        net = timm.create_model('efficientnet_b0', pretrained=False)
    else:
        sys.exit("wrong specification")
    net = nn.Sequential(*list(net.children())[:-5])
    net = truncate_efficientnet(net, trunc)

    test_input = test_input = torch.zeros(size=(1, 3, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(net(test_input).shape)
    
    
    fc = nn.Sequential(*[
                nn.Flatten(),
                nn.Linear(in_features=linear_input_dim,
                        out_features=num_classes,
                        bias=True)
            ])
    net = nn.Sequential(net, fc)
    return net
    

    


if __name__ == "__main__":
    pass

