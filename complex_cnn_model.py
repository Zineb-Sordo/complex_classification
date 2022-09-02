from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexLayers import ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu, complex_normalize, complex_avg_pool2d
from torch.nn.functional import dropout2d
import numpy as np
import pytorch_lightning as pl
from complex_activation_functions import zReLU, modReLU


# def center_crop(data, shape: Tuple[int, int]):
#
#     w_from = (data.shape[-2] - shape[0]) // 2
#     h_from = (data.shape[-1] - shape[1]) // 2
#     w_to = w_from + shape[0]
#     h_to = h_from + shape[1]
#
#     return data[..., w_from:w_to, h_from:h_to]


class ComplexPreActBlock(pl.LightningModule):
    """Pre-activation complex version of the BasicBlock."""

    expansion = 1

    def __init__(self, activation_function, in_planes, planes, stride=1,):
        super(ComplexPreActBlock, self).__init__()

        self.activation_function = activation_function
        self.Cbn1 = ComplexBatchNorm2d(in_planes, track_running_stats=False)
        self.Cconv1 = ComplexConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.Cbn2 = ComplexBatchNorm2d(planes)
        self.Cconv2 = ComplexConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                ComplexConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    )
            )

    def forward(self, x):
        if self.activation_function == "complex_relu":
            out = complex_relu(self.Cbn1(x))
        elif self.activation_function == "modReLU":
            out = modReLU(self.Cbn1(x), 0.1)
        elif self.activation_function == "zReLU":
            out = zReLU(self.Cbn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.Cconv1(out)
        if self.activation_function == "CReLU":
            out = self.Cconv2(complex_relu(self.Cbn2(out)))
        elif self.activation_function == "modReLU":
            out = self.Cconv2(modReLU(self.Cbn2(out), 0.1))
        elif self.activation_function == "zReLU":
            out = self.Cconv2(zReLU(self.Cbn2(out)))
        out += shortcut
        return out


def complex_dropout2d(input, p=0.5, training=True):
    if torch.cuda.is_available():
        mask = torch.ones(*input.shape, dtype=torch.float32, device=torch.device('cuda'))
    else:
        mask = torch.ones(*input.shape, dtype = torch.float32) #when using CPU only
    mask = dropout2d(mask, p, training) * 1 / (1-p)
    mask.type(input.dtype)
    return mask * input


class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout2d(input, self.p)
        else:
            return input


class ComplexPreActResNetFFTKnee(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            image_shape,
            activation_function,
            data_space,
            num_classes=4,
            drop_prob=0.5,
            return_features=False,
    ):
        super(ComplexPreActResNetFFTKnee, self).__init__()
        self.in_planes = 64
        self.activation_function = activation_function

        self.conv_comp = ComplexConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1_p = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, activation_function, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, activation_function, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, activation_function, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, activation_function, 512, num_blocks[3], stride=2)

        self.dropout = ComplexDropout2d(p=drop_prob)
        self.image_shape = image_shape
        self.data_space = data_space

        in_dim = 512 * block.expansion * 100
        batch_size = 8

        self.linear_mtear = nn.Linear(batch_size, num_classes)
        self.linear_acl = nn.Linear(batch_size, num_classes)
        self.linear_abnormal = nn.Linear(batch_size, num_classes)
        self.linear_cartilage = nn.Linear(batch_size, num_classes)

        self.Clinear_mtear = ComplexLinear(in_dim, num_classes)
        self.Clinear_acl = ComplexLinear(in_dim, num_classes)
        self.Clinear_abnormal = ComplexLinear(in_dim, num_classes)
        self.Clinear_cartilage = ComplexLinear(in_dim, num_classes)

        self.complexLinear = ComplexLinear(in_dim, num_classes)

    def _make_layer(self, block, activation_function, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(activation_function, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, kspace):
        # print("the kspace shape is {} and dtype is {}".format(kspace.shape, kspace.dtype)) # torch.size([8, 1, 640, 400])
        if self.data_space == 'complex_input':
            if torch.cuda.is_available():
                out = torch.complex(kspace.real, kspace.imag).cuda().type(torch.complex64)
            else:
                out = torch.complex(kspace.real, kspace.imag).type(torch.complex64)
            out = self.conv_comp(out)
        out = self.dropout(out)

        layer_1_out = self.layer1(out)
        # print("layer1 shape is {}".format(layer_1_out.shape))
        layer_2_out = self.layer2(layer_1_out)
        # print("layer2 shape is {}".format(layer_2_out.shape))
        layer_3_out = self.layer3(layer_2_out)
        # print("layer3 shape is {}".format(layer_3_out.shape))
        layer_4_out = self.layer4(layer_3_out)
        # print("layer4 shape is {}".format(layer_4_out.shape))
        out = complex_avg_pool2d(layer_4_out, 4)
        # print("complex_avg_pool2d shape is {}".format(out.shape))
        out = out.view(out.size(0), -1)
        # print("out.view shape is {}".format(out.shape))

        out_mtear = self.Clinear_mtear(out)
        out_acl = self.Clinear_acl(out)
        out_cartilage = self.Clinear_cartilage(out)
        out_abnormal = self.Clinear_abnormal(out)

        # First approach: output is magnitude
        out_mtear = out_mtear.abs()
        out_acl = out_acl.abs()
        out_cartilage = out_cartilage.abs()
        out_abnormal = out_abnormal.abs()

        # Second approach: output the stacked magnitude and phase

        # out_mtear = torch.stack((out_mtear.abs(), out_mtear.angle()), axis=1).float()
        # out_mtear = out_mtear.view(out_mtear.size(0), -1)
        #
        # out_acl = torch.stack((out_acl.abs(), out_acl.angle()), axis=1).float()
        # out_acl = out_acl.view(out_acl.size(0), -1)
        #
        # out_cartilage = torch.stack((out_cartilage.abs(), out_cartilage.angle()), axis=1).float()
        # out_cartilage = out_cartilage.view(out_cartilage.size(0), -1)
        #
        # out_abnormal = torch.stack((out_abnormal.abs(), out_abnormal.angle()), axis=1).float()
        # out_abnormal = out_abnormal.view(out_abnormal.size(0), -1)
        #
        # out_mtear = self.linear_mtear(out_mtear)
        # out_acl = self.linear_acl(out_acl)
        # out_cartilage = self.linear_cartilage(out_cartilage)
        # out_abnormal = self.linear_abnormal(out_abnormal)

        # Third approach is use a convolution of the magnitude and phase channels
        #print("outputs = {}, {}, {}, {}".format(out_abnormal, out_mtear, out_acl, out_cartilage))
        return out_abnormal, out_mtear, out_acl, out_cartilage


def complex_resnet18_knee(activation_function, image_shape, data_space, drop_prob=0.3, return_features=False):
    return ComplexPreActResNetFFTKnee(
        ComplexPreActBlock,
        [2, 2, 2, 2],
        drop_prob=drop_prob,
        image_shape=image_shape,
        data_space=data_space,
        activation_function=activation_function,
        return_features=return_features
    )

