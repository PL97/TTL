from hashlib import new
import torch
from torch import Tensor
import torch.nn as nn
# from .._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


NORMAL_BLOCK = 1
SLIMING_BLOCK = -1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        slim: int = 1
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        if slim == NORMAL_BLOCK or (slim == SLIMING_BLOCK and downsample != None):
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.conv3 = conv1x1(width, inplanes)
            self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        shrink_coefficient: int = 1, # 1 as default ResNet, can be changed to 2, 4, 8, etc.
        load_up_to: int = -1 # 4 as default ResNet, can be changed to 1, 2, 3 only
    ) -> None:
        # [3, 4, 6, 3]
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        out_planes_lst = [64, 128, 256, 512]
        self.load_up_to = load_up_to
        planes_idxs = np.cumsum(layers)
        

        new_out_planes_lst = []
        for idx in range(4):
            new_out_planes_lst.extend([out_planes_lst[idx]] * layers[idx])
        new_out_planes_lst = np.asarray(new_out_planes_lst)
        
        print("before slim:", new_out_planes_lst)
        # slim down the channels 
        if load_up_to != -1:
            new_out_planes_lst[load_up_to:] = new_out_planes_lst[load_up_to:]/shrink_coefficient

        # handle the bottleneck hetegentity (bottleneck1 differ from all other bottlenecks)
        slim_factors = np.ones(4).astype(int)
        if load_up_to == 0:
            slim_factors[0] = NORMAL_BLOCK
        else:
            for idx, val in enumerate(planes_idxs):
                if load_up_to < val:
                    slim_factors[idx] = SLIMING_BLOCK
                    slim_factors[idx+1:] = shrink_coefficient
                    break

        # print(slim_factors, load_up_to)

        print("after slim:", new_out_planes_lst)
  
        
        self.layer1 = self._make_layer(block, new_out_planes_lst[:planes_idxs[0]], layers[0], slim=slim_factors[0])
        self.layer2 = self._make_layer(block, new_out_planes_lst[planes_idxs[0]:planes_idxs[1]], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], slim=slim_factors[1])
        self.layer3 = self._make_layer(block, new_out_planes_lst[planes_idxs[1]:planes_idxs[2]], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], slim=slim_factors[2])
        self.layer4 = self._make_layer(block, new_out_planes_lst[planes_idxs[2]:planes_idxs[3]], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], slim=slim_factors[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if slim_factors[-1] != SLIMING_BLOCK:
            self.fc = nn.Linear(new_out_planes_lst[-1] * block.expansion, num_classes)
        else:
            # print(new_out_planes_lst[-1], block.expansion, shrink_coefficient)
            # print(new_out_planes_lst)
            # asdf
            self.fc = nn.Linear(new_out_planes_lst[-3] * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: list, blocks: int,
                    stride: int = 1, dilate: bool = False, slim: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes[0] * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes[0] * block.expansion, stride),
                norm_layer(planes[0] * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes[0], stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes[0] * block.expansion
        for idx in range(1, blocks):
            ## make sure the skip connect has channel matched
            # tmp_inplanes = planes[idx] * block.expansion         
            layers.append(block(self.inplanes, planes[idx], groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, slim=slim))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # model.load_state_dict(state_dict, strict=False)
        # weights = {k: v for k, v in weights.items() if k in lm_dict}
        # for k in model.state_dict().keys():
        for name, param in model.named_parameters():
            # param.data = nn.parameter.Parameter(torch.ones_like(param))
            if param.shape == state_dict[name].shape:
                # param = nn.parameter.Parameter(state_dict[name])
                model.state_dict()[name].data.copy_(state_dict[name])
                print(name, "load")
            else:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'bn' in name or ('downsample' in name and len(param.shape) == 1):
                    nn.init.ones_(param)
                elif 'weight' in name:
                    print(name, "kaiming", param.shape)
                    nn.init.kaiming_normal_(param)


        for name, param in model.named_parameters():
            param.requires_grad = True
            # if param.shape == state_dict[name].shape:
            #     print((param == state_dict[name]).all(), name)


    return model



def layer_wise_slim_resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)




if __name__ == "__main__":
    
    net = layer_wise_slim_resnet50(shrink_coefficient=2, load_up_to=14, num_classes=2, pretrained=True)
    print(net)
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
        else:
            print(name, "freeze")
    # print(net)
    # print(net)
    test_x = torch.randn(1, 3, 224, 224)
    from torchvision.models import resnet50
    # print(resnet50())
    # print(net)
    print(test_x.shape)
    print(net(test_x).shape)
    
    