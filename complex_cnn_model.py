import torch
import torch.nn as nn
import sys

from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexLayers import ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu, complex_normalize, complex_avg_pool2d
from torch.nn.functional import dropout2d
import pytorch_lightning as pl
from complex_activation_functions import zReLU, modReLU, cardioid


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

            out = modReLU(self.Cbn1(x), nn.Parameter(torch.Tensor(x.shape)).cuda())
        elif self.activation_function == "zReLU":
            out = zReLU(self.Cbn1(x))
        elif self.activation_function == "cardioid":
            out = cardioid(self.Cbn1(x))

        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.Cconv1(out)
        if self.activation_function == "CReLU":
            out = self.Cconv2(complex_relu(self.Cbn2(out)))
        elif self.activation_function == "zReLU":
            out = self.Cconv2(zReLU(self.Cbn2(out)))
        elif self.activation_function == "cardioid":
            out = self.Cconv2(cardioid(self.Cbn2(out)))

        elif self.activation_function == "modReLU":
            out = modReLU(self.Cbn2(out), nn.Parameter(torch.Tensor(out.shape)).cuda())

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
            output_type,
            num_classes=2,
            drop_prob=0.5,
            return_features=False,
    ):
        super(ComplexPreActResNetFFTKnee, self).__init__()
        self.in_planes = 64
        self.activation_function = activation_function

        self.conv_comp = ComplexConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1d_2 = ComplexBatchNorm2d(64)


        self.layer1 = self._make_layer(block, activation_function, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, activation_function, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, activation_function, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, activation_function, 512, num_blocks[3], stride=2)

        self.dropout = ComplexDropout2d(p=drop_prob)
        self.image_shape = image_shape
        self.data_space = data_space
        self.output_type = output_type

        in_dim = 512 * block.expansion * 100
        out_dim = 1024

        self.Clinear_mtear = ComplexLinear(in_dim, num_classes)
        self.Clinear_acl = ComplexLinear(in_dim, num_classes)
        self.Clinear_abnormal = ComplexLinear(in_dim, num_classes)
        self.Clinear_cartilage = ComplexLinear(in_dim, num_classes)

        self.Clinear = ComplexLinear(in_dim, out_dim)
        self.bn1d = ComplexBatchNorm1d(out_dim)

        # Linear layers for mag only
        self.linear_mtear = nn.Linear(out_dim, num_classes)
        self.linear_acl = nn.Linear(out_dim, num_classes)
        self.linear_abnormal = nn.Linear(out_dim, num_classes)
        self.linear_cartilage = nn.Linear(out_dim, num_classes)

        # Linear layers for mag + phase
        self.linear_mtear_2 = nn.Linear(2*out_dim, num_classes)
        self.linear_acl_2 = nn.Linear(2*out_dim, num_classes)
        self.linear_abnormal_2 = nn.Linear(2*out_dim, num_classes)
        self.linear_cartilage_2 = nn.Linear(2*out_dim, num_classes)

    def _make_layer(self, block, activation_function, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(activation_function, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, kspace):

        # print("the kspace shape is {} and dtype is {}".format(kspace.shape, kspace.dtype)) # torch.size([8, 1, 640, 400])
        if torch.cuda.is_available():
            out = torch.complex(kspace.real, kspace.imag).cuda().type(torch.complex64)
        else:
            out = torch.complex(kspace.real, kspace.imag).type(torch.complex64)
        out = self.conv_comp(out)

        if self.activation_function == "complex_relu":
            out = complex_relu(self.bn1d_2(out))
        elif self.activation_function == "modReLU":
            out = modReLU(self.bn1d_2(out), nn.Parameter(torch.Tensor(out.shape)).cuda())
        elif self.activation_function == "zReLU":
            out = zReLU(self.bn1d_2(out))
        elif self.activation_function == "cardioid":
            out = cardioid(self.bn1d_2(out))
        out = self.dropout(out)

        layer_1_out = self.layer1(out) # [8,64,320,320]
        layer_2_out = self.layer2(layer_1_out) # [8,128,320,320]
        layer_3_out = self.layer3(layer_2_out) # [8,256,320,320]
        layer_4_out = self.layer4(layer_3_out) # [8,512,320,320]
        out = complex_avg_pool2d(layer_4_out, 4) # [8,512,10,10]
        out = out.view(out.size(0), -1)  # [8,51200]
        out = self.Clinear(out) # [8,1024]

        if self.activation_function == "complex_relu":
            out = complex_relu(self.bn1d(out))
        elif self.activation_function == "modReLU":
            out = modReLU(self.bn1d(out), nn.Parameter(torch.Tensor(out.shape)).cuda())
        elif self.activation_function == "zReLU":
            out = zReLU(self.bn1d(out))
        elif self.activation_function == "cardioid":
            out = cardioid(self.bn1d(out))
        out = self.dropout(out)

        if self.output_type == "mag_phase":

            out = torch.stack((out.abs(), out.angle()), axis=1).float() # [8,2,2048]
            out = out.view(out.size(0), -1) # [8,2048]

            out_mtear = self.linear_mtear_2(out) # [8, 2]
            out_acl = self.linear_acl_2(out) # [8, 2]
            out_cartilage = self.linear_cartilage_2(out) # [8, 2]
            out_abnormal = self.linear_abnormal_2(out) # [8, 2]

        elif self.output_type == "magnitude":
            out = out.abs()

            out_mtear = self.linear_mtear(out)  # [8, 2]
            out_acl = self.linear_acl(out)  # [8, 2]
            out_cartilage = self.linear_cartilage(out)  # [8, 2]
            out_abnormal = self.linear_abnormal(out)  # [8, 2]
        else:
            print("Error in type of output selected")
            sys.exit()

        return out_abnormal, out_mtear, out_acl, out_cartilage


def complex_resnet18_knee(activation_function, image_shape, data_space, output_type, drop_prob=0.5, return_features=False):
    return ComplexPreActResNetFFTKnee(
        ComplexPreActBlock,
        [2, 2, 2, 2],
        drop_prob=drop_prob,
        image_shape=image_shape,
        data_space=data_space,
        output_type=output_type,
        activation_function=activation_function,
        return_features=return_features
    )

