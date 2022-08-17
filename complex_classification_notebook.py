#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import h5py 

get_ipython().run_line_magic('matplotlib', 'inline')

import math
import os
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import pytorch_lightning as pl
import torch
from joblib import dump, load
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset
from tqdm.auto import tqdm

from fastmri.fftc import ifft2c_new


# classifier.py imports 
import argparse
import os
import pathlib
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from tqdm.auto import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import sys
import inspect

import fire
import itertools


# In[2]:


class MultiDataset(Dataset):
    def __init__(self, split_csv_file: str, mode: str, dev_mode: bool = False):
        super().__init__()
        # read csv file with filenames
        self.split_csv_file = Path(split_csv_file)
        assert self.split_csv_file.is_file()

        self.metadata = pd.read_csv(self.split_csv_file)
        if dev_mode:
            self.metadata = self.metadata.iloc[:1500]

        assert "data_split" in self.metadata.columns
        assert "location" in self.metadata.columns

        metadata_grouped = self.metadata.groupby("data_split")
        self.metadata_by_mode = {
            e: metadata_grouped.get_group(e) for e in metadata_grouped.groups
        }

        self.mode = mode
        assert mode in self.metadata_by_mode

    def __len__(self):
        assert self.mode in self.metadata_by_mode
        return self.metadata_by_mode[self.mode].shape[0]

    def get_metadata_value(self, index, key):
        assert self.mode in self.metadata_by_mode
        assert key in self.metadata_by_mode[self.mode].iloc[index]
        return self.metadata_by_mode[self.mode].iloc[index][key]

    def __getitem__(self, index):
        raise NotImplementedError


# In[ ]:





# In[ ]:


class KneeDataset(MultiDataset):
    def __init__(
        self, 
        split_csv_file: str,
        mode: str,
        label_type: str,
        data_space: str,
        coil_type='sc'
    ):
        super().__init__(split_csv_file=split_csv_file, mode=mode)
        fields = [
            "volume_id",
            "slice_id",
            "sc_kspace",
            "mc_kspace",
            "recon_esc",
            "recon_rss",
            "label",
            "data_split",
            "dataset",
            "location",
            "max_value",
        ]
        
        self.coil_type = coil_type
        assert self.coil_type in {"sc", "mc"}
        self.label_type = label_type
        self.data_space = data_space
        self.mode = mode
        
    def parse_label(self, label_arr: Sequence[str]) -> torch.Tensor:
        label_arr = label_arr.replace("[", "").replace("]", "").replace("'", "")
        label_arr = label_arr.split(",")
        
        if "None" in label_arr:
            return torch.Tensor([0.0, 0.0, 0.0, 0.0]).float()

        new_labels = []

        for label in label_arr:
            if "ACL" in label:
                new_labels.append(1)
            elif "Meniscus Tear" in label:
                new_labels.append(2)
            elif "cartilage" in label.lower():
                new_labels.append(3)
            else:
                new_labels.append(4)

        # Abnormal being any other pathology different from ACL, Mtear or cartilage
        abnormal = 1.0 if 4 in new_labels else 0.0
        cartilage = 1.0 if 3 in new_labels else 0.0
        mtear = 1.0 if 2 in new_labels else 0.0
        acl = 1.0 if 1 in new_labels else 0.0

        return torch.Tensor(np.array([abnormal, mtear, acl, cartilage])).float()

    def __getitem__(self, index):
        assert self.mode in self.metadata_by_mode
        loc = self.get_metadata_value(index, "location")

        info = self.metadata_by_mode[self.mode].iloc[index]
        kspace_key = "sc_kspace" if self.coil_type == "sc" else "mc_kspace"
        target_key = "recon_esc" if self.coil_type == "sc" else "recon_rss"

        with h5py.File(loc) as f:
            kspace_data = f[kspace_key][:]
            target_data = f[target_key][:]

            image_data = torch.from_numpy(kspace_data)
            image_data = ifft2c_new(image_data)
            kspace_data = torch.view_as_complex(image_data)

            parameters = {
                kspace_key: kspace_data,
                target_key: target_data,
                "volume_id": info.volume_id,
                "slice_id": info.slice_id,
                "data_split": info.data_split,
                "dataset": info.dataset,
                "location": info.location,
            }
            if self.label_type == "knee_multilabel":
                parameters["label"] = self.parse_multilabel(info.labels)
            elif self.label_type == "knee":
                parameters["label"] = self.parse_label(info.labels)
            else:
                raise NotImplementedError(
                    f"Label type {self.label_type} not implemented"
                )                

        sample = self.sample_template(**parameters)
        return sample

        


# In[ ]:


class KneeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        split_csv_file: str,
        label_type: str,
        coil_type: str,
        batch_size: int,
        data_space: str,
        sampler_filename: Optional[str] = None,
        combine_class_recon: bool = False,
        dev_mode: bool = False,
        num_workers: int = 0,
    ):
        super().__init__()

        self.split_csv_file = split_csv_file
        self.coil_type = coil_type
        self.batch_size = batch_size
        self.sampler_filename = sampler_filename
        self.dev_mode = dev_mode
        self.num_workers = num_workers
        self.data_space = data_space
        self.label_type = label_type
    
    def setup(self, stage: Optional[str] = None):
        # get data split names
        test_mode: Optional[str]
        train_mode = "train_class"
        val_mode = "val_class"
        test_mode = "test_class"

        # initialize datasets
        self.train_dataset = KneeDataset(
            split_csv_file=self.split_csv_file,
            coil_type=self.coil_type,
            mode=train_mode,
            label_type=self.label_type,
            data_space=self.data_space,
        )
        self.val_dataset = KneeDataset(
            split_csv_file=self.split_csv_file,
            coil_type=self.coil_type,
            mode=val_mode,
            label_type=self.label_type,
            data_space=self.data_space,
        )

        self.test_dataset = KneeDataset(
                split_csv_file=self.split_csv_file,
                mode=test_mode,
                coil_type=self.coil_type,
                label_type=self.label_type,
                data_space=self.data_space,
        )
    
    if not os.path.exists(self.sampler_filename):
        raise ValueError("Weighted sampler does not exist")
    assert Path(self.sampler_filename).is_file()
        # load the sampler
    self.train_sampler = load(self.sampler_filename)
    
    def train_dataloader(self) -> DataLoader:
                return DataLoader(
                        self.train_dataset,
                        batch_size=self.batch_size,
                        sampler=self.train_sampler,
                )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


# ### Implement the complex network

# In[ ]:


from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexLayers import ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu, complex_normalize, complex_avg_pool2d
from torch.nn.functional import dropout2d
#from complexPyTorch.complexLayers import ComplexDropout2d
import numpy as np


def center_crop(data, shape: Tuple[int, int]):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


class ComplexPreActBlock(nn.Module):
    """Pre-activation complex version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ComplexPreActBlock, self).__init__()
        self.Cbn1 = ComplexBatchNorm2d(in_planes, track_running_stats=False)
        self.Cconv1 = ComplexConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.Cbn2 = ComplexBatchNorm2d(planes)
        self.Cconv2 = ComplexConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
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
        out = complex_relu(self.Cbn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.Cconv1(out)
        out = self.Cconv2(complex_relu(self.Cbn2(out)))
        out += shortcut
        return out
    
def complex_dropout2d(input, p=0.5, training=True):
    mask = torch.ones(*input.shape, dtype = torch.float32, device=torch.device('cuda'))
    #mask = torch.ones(*input.shape, dtype = torch.float32)

    mask = dropout2d(mask, p, training)*1/(1-p)
    mask.type(input.dtype)
    return mask*input
    
class ComplexDropout2d(nn.Module):
    def __init__(self,p=0.5):
        super(ComplexDropout2d,self).__init__()
        self.p = p

    def forward(self,input):
        if self.training:
            return complex_dropout2d(input,self.p)
        else:
            return input


class ComplexPreActResNetFFT_Knee(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            image_shape,
            data_space,
            num_classes=4,
            drop_prob=0.5,
            return_features=False,
    ):
        super(ComplexPreActResNetFFT_Knee, self).__init__()
        self.in_planes = 64

        self.conv_comp = ComplexConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.conv1_p = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        
        self.dropout = ComplexDropout2d(p=drop_prob)
        self.image_shape = image_shape
        self.data_space = data_space
        
        in_dim = 512 * block.expansion * 100
        
        self.linear_mtear = nn.Linear(8, num_classes)
        self.linear_acl = nn.Linear(8,num_classes)
        self.linear_abnormal = nn.Linear(8, num_classes)
        self.linear_cartilage = nn.Linear(8, num_classes)
        
        self.Clinear_mtear = ComplexLinear(in_dim, num_classes)
        self.Clinear_acl = ComplexLinear(in_dim,num_classes)
        self.Clinear_abnormal = ComplexLinear(in_dim, num_classes)
        self.Clinear_cartilage = ComplexLinear(in_dim, num_classes)
        
  
        self.complexLinear = ComplexLinear(in_dim, num_classes)
        #self.bn1d = ComplexBatchNorm1d(out_dim, track_running_stats = False)
        #self.complexLinear2 = ComplexLinear(out_dim, num_classes)
        
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, kspace):
        print(kspace.shape) # torch.size([8, 1, 640, 400])
        if self.data_space == 'complex_input':
            out = torch.complex(kspace.real, kspace.imag).cuda().type(torch.complex64)
            out = center_crop(out, self.image_shape)
            out = self.conv_comp(out)
            
        out = self.dropout(out)

        layer_1_out = self.layer1(out)
        layer_2_out = self.layer2(layer_1_out)
        
        layer_3_out = self.layer3(layer_2_out)
        layer_4_out = self.layer4(layer_3_out)
        print(layer_4_out.shape)
        out = complex_avg_pool2d(layer_4_out, 4)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.dropout(out)
        out = complex_relu(out)
        

        # First approach: output is magnitude 
        out_mtear = self.Clinear_mtear(out)
        out_acl = self.Clinear_acl(out)
        out_cartilage = self.Clinear_cartilage(out)
        out_abnormal = self.Clinear_abnormal(out)
        
        #out_mtear = out_mtear.abs()
        #out_acl = out_acl.abs()
        #out_cartilage = out_cartilage.abs()
        #out_abnormal = out_abnormal.abs()
        
        
        # Second approach: output the stacked magnitude and phase
      
        out_mtear = torch.stack((out_mtear.abs(), out_mtear.angle()), axis=1).float()
        out_mtear = out_mtear.view(out_mtear.size(0),-1)

        out_acl = torch.stack((out_acl.abs(), out_acl.angle()), axis=1).float()
        out_acl = out_acl.view(out_acl.size(0),-1)

        out_cartilage = torch.stack((out_cartilage.abs(), out_cartilage.angle()), axis=1).float()
        out_cartilage = out_cartilage.view(out_cartilage.size(0),-1)

        out_abnormal = torch.stack((out_abnormal.abs(), out_abnormal.angle()), axis=1).float()
        out_abnormal = out_abnormal.view(out_abnormal.size(0),-1)

        out_mtear = self.linear_mtear(out_mtear)
        out_acl = self.linear_acl(out_acl)
        out_cartilage = self.linear_cartilage(out_cartilage)
        out_abnormal = self.linear_abnormal(out_abnormal)
        
        # Third approach is use a convolution of the magnitude and phase channels 
        
        
        print("outputs = {}, {}, {}, {}".format(out_abnormal, out_mtear, out_acl, out_cartilage))
        return out_abnormal, out_mtear, out_acl, out_cartilage



def ComplexPreActResNet18FFT_Knee(image_shape, data_space, drop_prob=0.5, return_features=False):
    return ComplexPreActResNetFFT_Knee(
        ComplexPreActBlock,
        [2, 2, 2, 2],
        drop_prob=drop_prob,
        image_shape=image_shape,
        data_space=data_space,
        return_features=return_features
    )


def test():
    net = ComplexPreActResNet18FFT_Knee(drop_out=0.5)
    y = net((torch.randn(1, 3, 32, 32)))
    print(y.size())


# ### Classification metrics 
# 

# In[ ]:


from typing import Dict
from sklearn import metrics
import torch
from torchmetrics import functional
import numpy as np


def compute_auc(preds, labels):
    return functional.auroc(preds=preds, target=labels, num_classes=2)


def compute_accuracy(preds, labels):
    return functional.accuracy(preds=preds, target=labels)


def get_operating_point(preds, labels, operating_point=None, threshold=0.1):
    # print(preds)
    preds = preds.cpu()
    preds_positive = preds[:, 1].numpy()

    labels = labels.cpu()

    if operating_point is None:
        fpr, tpr, thresholds = metrics.roc_curve(labels.numpy(), preds_positive)
        operating_point = thresholds[fpr > 0.25][0]

        try:
            fnr = 1 - tpr
            operating_point = thresholds[fnr < threshold][0]
        except IndexError:
            operating_point = thresholds[fpr > 0.25][0]


    test_predictions = (preds_positive > operating_point).astype(int)

    sensitivity = metrics.recall_score(labels, test_predictions)
    specificity = metrics.recall_score(
        np.abs(1 - labels.numpy()), np.abs(1 - test_predictions)
    )
    balanced_acc = metrics.balanced_accuracy_score(labels.numpy(), test_predictions)

    print(f"b acc = {balanced_acc}, specificity = {specificity}, sensitivity = {sensitivity}")
    return balanced_acc, specificity, sensitivity, operating_point


def evaluate_classifier(
    preds: torch.Tensor, labels: torch.Tensor, operating_point: float = None
) -> Dict:
    try:
        auc = compute_auc(preds=preds[:, 1], labels=labels).item()
        balanced_acc, specificity, sensitivity, operating_point = get_operating_point(
            preds, labels, operating_point=operating_point
        )
    except ValueError:
        auc = np.nan
        specificity = np.nan
        sensitivity = np.nan
        balanced_acc = np.nan
        operating_point = np.nan
        print(
            "No negative samples in targets, false positive value should be meaningless"
        )

    return dict(
        auc=auc,
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_accuracy=balanced_acc,
        operating_point=operating_point,
    )


def get_bootstrap_estimates(
    preds: torch.Tensor,
    labels: torch.Tensor,
    operating_point: float,
    n_bootstrap_samples: int,
) -> Dict:
    N = len(preds)

    arr_auc = []
    arr_sensitivity = []
    arr_specificity = []
    arr_balanced_accuracy = []

    metrics = evaluate_classifier(
        preds=preds, labels=labels, operating_point=operating_point,
    )
    auc = metrics["auc"]
    sensitivity = metrics["sensitivity"]
    specificity = metrics["specificity"]
    balanced_accuracy = metrics["balanced_accuracy"]

    for i in range(n_bootstrap_samples):
        bootstrap_index = np.random.choice(
            np.arange(N), size=N, replace=True, p=np.ones(N) / N
        )
        bootstrap_preds = preds[bootstrap_index]
        bootstrap_labels = labels[bootstrap_index]

        if bootstrap_labels.sum() == 0:
            bootstrap_index = np.random.choice(
                np.arange(N), size=N, replace=True, p=np.ones(N) / N
            )

        bootstrap_metrics = evaluate_classifier(
            preds=bootstrap_preds,
            labels=bootstrap_labels,
            operating_point=operating_point,
        )
        arr_auc.append(bootstrap_metrics["auc"])
        arr_sensitivity.append(bootstrap_metrics["sensitivity"])
        arr_specificity.append(bootstrap_metrics["specificity"])
        arr_balanced_accuracy.append(bootstrap_metrics["balanced_accuracy"])

    std_auc = np.array(arr_auc).std()
    std_sensitivity = np.array(arr_sensitivity).std()
    std_specificity = np.array(arr_specificity).std()
    std_balanced_accuracy = np.array(arr_balanced_accuracy).std()

    print(operating_point)

    return dict(
        auc=auc,
        std_auc=std_auc,
        sensitivity=sensitivity,
        std_sensitivity=std_sensitivity,
        specificity=specificity,
        std_specificity=std_specificity,
        balanced_accuracy=balanced_accuracy,
        std_balanced_accuracy=std_balanced_accuracy,
        operating_point=operating_point,
    )


def classifier_metrics(
    val_preds: torch.Tensor,
    val_labels: torch.Tensor,
    test_preds: torch.Tensor,
    test_labels: torch.Tensor,
    threshold: float,
    n_bootstrap_samples: int,
):
    _, _, _, operating_point = get_operating_point(
        preds=val_preds, labels=val_labels, operating_point=None, threshold=threshold
    )

    return (
        operating_point,
        get_bootstrap_estimates(
            preds=test_preds, labels=test_labels, operating_point=operating_point
        ),
    )


# In[ ]:


# Here write the RSS class called complex_resnet here

import os
from typing import Tuple, Optional
from torch.serialization import validate_cuda_device
from tqdm.auto import tqdm
import numpy as np
from copy import deepcopy
import pytorch_lightning as pl

import torch
import torch.optim as optim
import torch.nn as nn

from metrics.classification_metrics import (
    compute_accuracy,
    get_operating_point,
    evaluate_classifier,
    get_bootstrap_estimates,
)
from torch.utils.data import DataLoader


def get_model(
    data_type: str,
    model_type: str,
    drop_prob: float,
    data_space: str,
    image_shape: Tuple[int, int],
    sequences: Optional[Tuple[str, str, str]] = None,
    return_features=False,
    num_labels=4
) -> nn.Module:
    if data_type == "knee":
        if model_type == "complex_preact_resnet18":
            return ComplexPreActResNet18FFT_Knee(image_shape=image_shape, drop_prob=drop_prob, data_space=data_space, return_features=return_features)
    else:
        raise NotImplementedError(f"Model type {model_type} not complex and not implemented")


class RSS(pl.LightningModule):
    def __init__(self,
                 model_type: str,
                 data_type: str,
                 drop_prob: float,
                 kspace_shape: Tuple[int, int],
                 image_shape: Tuple[int, int],
                 device: torch.device,
                 data_space: str,
                 #label_names: str,
                 coil_type: str = "sc",
                 lr: float = 1e-5,
                 weight_decay: float = 1e-5,
                 lr_gamma: float = 0.1,
                 lr_step_size: int = 20,
                 # dwi_kspace_shape: Optional[Tuple[int, int]] = None,
                 num_labels = 4,
                 n_bootstrap_samples: int = 50,
                 sequences: Optional[Tuple[str, str, str]] = ["t2", "b50"],
                 return_features: str = False
                ):
        super().__init__()
        self.save_hyperparameters()

        # data and task type
        self.data_type = data_type
        self.image_shape = image_shape

        # model type and parameters
        self.model_type = model_type
        self.drop_prob = drop_prob
        #self.label_names = label_names
        self.num_labels = num_labels

        # optimizer parameters
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_step_size = lr_step_size
        self.weight_decay = weight_decay

        # loss function
        self.criterion = nn.CrossEntropyLoss()
        self.kspace_shape = kspace_shape

        self.sequences = sequences
        self.data_space = data_space
        self.return_features = return_features

        # get model depending on data and model type
        self.model = get_model(
            data_type=self.data_type,
            model_type=self.model_type,
            drop_prob=self.drop_prob,
            image_shape=self.image_shape,
            sequences=self.sequences,
            data_space=self.data_space,
            return_features=self.return_features,
        )

        self.val_operating_point = None
        self.n_bootstrap_samples = n_bootstrap_samples

    def forward(self, batch):
        kspace = batch.sc_kspace
        kspace = kspace.cuda().type(torch.complex64)
        return self.model(kspace.unsqueeze(1))

    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(preds, labels)

    def compute_loss_and_metrics(self, preds, labels, label_names):
    
        assert len(label_names) == self.num_labels
            
        pred_out, label_out = [], [] # To store preds and labels for each label
        acc_per_label = []

        loss = None
        #print("num labels:", self.num_labels)
        for i in range(0, self.num_labels):
            
            curr_loss = self.loss_fn(preds=preds[i], labels=labels[:,i])
            if loss is None:
                loss = curr_loss
            else:
                loss += curr_loss
                
            acc = compute_accuracy(preds[i], labels[:, i])
            acc_per_label.append(acc)
            
            self.log(label_names[i], acc, prog_bar=True)
            
        return loss

    def training_step(self, batch, batch_idx):
        labels = batch.label.long()
        
        # get predictions
        preds = self.forward(batch=batch)
        # print("preds shape: ",preds.shape)
        if self.data_type == "knee":
            labels_abnormal = labels[:, 0]
            labels_mtear = labels[:, 1]
            labels_acl = labels[:, 2]
            labels_cartilage = labels[:, 3]

            preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds
            loss = self.loss_fn(preds=preds_abnormal, labels=labels_abnormal)
            loss += self.loss_fn(preds=preds_mtear, labels=labels_mtear)
            loss += self.loss_fn(preds=preds_acl, labels=labels_acl)
            loss += self.loss_fn(preds=preds_cartilage, labels=labels_cartilage)
            
            acc_abnormal = compute_accuracy(preds_abnormal.max(1)[1], labels_abnormal)
            acc_mtear = compute_accuracy(preds_mtear.max(1)[1], labels_mtear)
            acc_acl = compute_accuracy(preds_acl.max(1)[1], labels_acl)
            acc_cartilage = compute_accuracy(preds_cartilage.max(1)[1], labels_cartilage)

            self.log("train_abnormal_acc", acc_abnormal, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_mtear_acc", acc_mtear, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_acl_acc", acc_acl, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train_acl_cartilage", acc_cartilage, prog_bar=True, on_step=True, on_epoch=True)
        
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        labels = batch.label.long()
        # get predictions
        preds = self.forward(batch=batch)
        #print("preds shape: ",preds.shape)
        if self.data_type == "knee":
            labels_abnormal = labels[:, 0]
            labels_mtear = labels[:, 1]
            labels_acl = labels[:, 2]
            labels_cartilage = labels[:, 3]

            preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds

            loss = self.loss_fn(preds=preds_abnormal, labels=labels_abnormal)
            loss += self.loss_fn(preds=preds_mtear, labels=labels_mtear)
            loss += self.loss_fn(preds=preds_acl, labels=labels_acl)
            loss += self.loss_fn(preds=preds_cartilage, labels=labels_cartilage)

        batch_size = labels.shape[0]
        return {
            "loss": loss,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "labels": labels,
            "preds": preds,
        }

    def collate_results(self, logs: Tuple) -> Tuple:
        loss = []
        loss_list = []
        n_sample_points = 0

        if self.data_type == "knee":
            labels_abnormal, labels_mtear, labels_acl, labels_cartilage = [], [], [], []
            preds_abnormal, preds_mtear, preds_acl, preds_cartilage = [], [], [], []

            for log_t in logs:
                loss.append(log_t["loss"].cpu() * log_t["batch_size"])
                n_sample_points += log_t["batch_size"]
                preds_t, labels_t = log_t["preds"], log_t["labels"]

                labels_abnormal.append(labels_t[:, 0])
                labels_mtear.append(labels_t[:, 1])
                labels_acl.append(labels_t[:, 2])
                labels_cartilage.append(labels_t[:, 3])

                preds_abnormal.append(preds_t[0])
                preds_mtear.append(preds_t[1])
                preds_acl.append(preds_t[2])
                preds_cartilage.append(preds_t[3])

            labels_abnormal = torch.cat(labels_abnormal, dim=0)
            labels_mtear = torch.cat(labels_mtear, dim=0)
            labels_acl = torch.cat(labels_acl, dim=0)
            labels_cartilage = torch.cat(labels_cartilage, dim=0)

            preds_abnormal = torch.cat(preds_abnormal, dim=0)
            preds_mtear = torch.cat(preds_mtear, dim=0)
            preds_acl = torch.cat(preds_acl, dim=0)
            preds_cartilage = torch.cat(preds_cartilage, dim=0)

            labels = [labels_abnormal, labels_mtear, labels_acl, labels_cartilage]
            preds = [preds_abnormal, preds_mtear, preds_acl, preds_cartilage]

            loss = np.sum(loss) / n_sample_points

            return [preds, labels, loss]

    def validation_epoch_end(self, val_logs):
        preds, labels, loss = self.collate_results(val_logs)
        self.log("val_loss", loss, prog_bar=True)

        if self.data_type == "knee":
            labels_abnormal, labels_mtear, labels_acl, labels_cartilage = labels
            preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds
            # print(preds_abnormal[0:10], preds_mtear[0:10], preds_acl[0:10])

            eval_metrics = {}
            eval_metrics["abnormal"] = evaluate_classifier(
                preds_abnormal, labels_abnormal
            )
            eval_metrics["mtear"] = evaluate_classifier(preds_mtear, labels_mtear)
            eval_metrics["acl"] = evaluate_classifier(preds_acl, labels_acl)
            eval_metrics["cartilage"] = evaluate_classifier(preds_cartilage, labels_cartilage)

            avg_auc = 0.0
            self.val_operating_point = {}
            keys = ["abnormal", "mtear", "acl", "cartilage"]
            for key in keys:
                key_score = eval_metrics[key]["auc"]
                self.log(f"val_auc_{key}", key_score, prog_bar=True)
                self.log(
                    f"val_bac_{key}",
                    eval_metrics[key]["balanced_accuracy"],
                    prog_bar=True,
                )
                avg_auc += key_score / len(keys)

                self.val_operating_point[key] = eval_metrics[key]["operating_point"]

            self.log(f"val_auc_mean", avg_auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        labels = batch.label.long()
        # get predictions
        preds = self.forward(batch=batch)

        if self.data_type == "knee":
            labels_abnormal = labels[:, 0]
            labels_mtear = labels[:, 1]
            labels_acl = labels[:, 2]
            labels_cartilage = labels[:, 3]

            preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds

            loss = self.loss_fn(preds=preds_abnormal, labels=labels_abnormal)
            loss += self.loss_fn(preds=preds_mtear, labels=labels_mtear)
            loss += self.loss_fn(preds=preds_acl, labels=labels_acl)
            loss += self.loss_fn(preds=preds_cartilage, labels=labels_cartilage)

        batch_size = labels.shape[0]
        return {
            "loss": loss,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "labels": labels,
            "preds": preds,
        }

    def test_epoch_end(self, test_logs):

        preds, labels, loss = self.collate_results(test_logs)
        self.log("test_loss", loss, prog_bar=True)

        if self.data_type == "knee":
            labels_abnormal, labels_mtear, labels_acl, labels_cartilage = labels
            preds_abnormal, preds_mtear, preds_acl, preds_cartilage = preds

            eval_metrics = {}
            eval_metrics["abnormal"] = evaluate_classifier(
                preds_abnormal,
                labels_abnormal,
                operating_point=self.val_operating_point["abnormal"],
            )
            eval_metrics["mtear"] = evaluate_classifier(
                preds_mtear,
                labels_mtear,
                operating_point=self.val_operating_point["mtear"],
            )
            eval_metrics["acl"] = evaluate_classifier(
                preds_acl, labels_acl,
                operating_point=self.val_operating_point["acl"],
            )
            eval_metrics["cartilage"] = evaluate_classifier(
                preds_cartilage, labels_cartilage,
                operating_point=self.val_operating_point["cartilage"],
            )
           
            avg_auc = 0.0
            test_operating_point = {}
            keys = ["abnormal", "mtear", "acl", "cartilage"]
            loss = 0
            prefix = f"test"
            for key in ["abnormal", "mtear", "acl", "cartilage"]:
                for metric in ["auc",
                                "sensitivity",
                                "specificity",
                                "balanced_accuracy",
                                "operating_point"]:
                    key_score = eval_metrics[key][metric]
                    self.log(
                        f"{prefix}_{key}_{metric}",
                        eval_metrics[key][metric],
                        prog_bar=True,
                     )
            
        return loss, eval_metrics

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )

        return [optimizer], [scheduler]




# ### Get the data and train the model

# In[ ]:


# Get the arguments 
def get_args():
    
    parser = argparse.ArgumentParser(description="Indirect MR Screener training")
    
    # logging parameters
    parser.add_argument("--model_dir", type=str, default="../trained_models")
    parser.add_argument("--log_dir", type=str, default="../trained_logs")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dev_mode", action="store_true")
    
    # data parameters
    parser.add_argument("--data_type", type=str, default="knee",)
    parser.add_argument("--data_space", type=str, default="complex_input")
    #parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--image_shape", type=int, default=[320, 320], nargs=2, required=False)
    parser.add_argument("--image_type", type=str, default='orig', required=False, choices=["orig"])
    
    # parser.add_argument("--split_csv_file", type=str, default='..//metadata_knee.csv', required=False)
    parser.add_argument("--split_csv_file", 
                        type=str, 
                        default='/gpfs/data/chopralab/zineb_project/fastMRI/MRI/data_processing/knee/metadata_knee.csv',
                        required=False)
    parser.add_argument("--recon_model_ckpt", type=str)
    parser.add_argument("--recon_model_type", type=str, default=["rss"], required=False, choices=["rss"])
    parser.add_argument("--mask_type", type=str, default="none")
    parser.add_argument("--k_fraction", type=float, default=0.25)
    parser.add_argument("--center_fraction", type=float, default=0.08)
    parser.add_argument("--coil_type", type=str, default="sc", choices=["sc", "mc"])

    parser.add_argument("--sampler_filename", type=str, default="../data_processing/knee/sampler_knee_tr.p")
    parser.add_argument(
        "--model_type",
        type=str,
        default="complex_preact_resnet18",
        choices=["complex_preact_resnet18",
                 "complex_preact_resnet50"
                ],
    )

    # training parameters
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=5)

    parser.add_argument("--n_masks", type=int, default=100)

    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--sweep_step", type=int)
    parser.add_argument('--debug',  default=True)

    args, unkown = parser.parse_known_args()
    
    return args


# In[ ]:


# get the data 
def get_data(args: argparse.ArgumentParser) -> pl.LightningDataModule:
    # get datamodule
    if args.data_type == "knee":
        # load mc data to obtain rss images
        datamodule = KneeDataModule(
            label_type="knee",
            split_csv_file=args.split_csv_file,
            coil_type=args.coil_type,
            batch_size=args.batch_size,
            sampler_filename=args.sampler_filename,
            data_space=args.data_space,
    )
    else:
        raise NotImplementedError

    return datamodule


# In[ ]:


# get the model 
def get_model(args: argparse.ArgumentParser, device: torch.device) -> pl.LightningModule:
    if args.data_type == "knee":
        model = complex_resnet(
            model_type=args.model_type,
            data_type=args.data_type,
            image_shape=[320, 320],
            drop_prob=args.drop_prob,
            kspace_shape=[640, 400],
            data_space=args.data_space,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_gamma=args.lr_gamma,
            lr_step_size=args.lr_step_size,
        )
    else:
        raise NotImplementedError
    return model
    


# In[ ]:


# train the model 

def train_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:
    log_dir = (
        Path(args.log_dir)
        / args.data_type
        / args.data_space
    )
    model_dir = str(args.model_dir) + '/' + args.data_space + '/' + str(args.n_seed)

    if not os.path.isdir(str(log_dir)):
        os.makedirs(str(log_dir))
    if not os.path.isdir(str(model_dir)):
        os.makedirs(str(model_dir))

    csv_logger = CSVLogger(save_dir=log_dir, name="Trained_Complex_CNN", version=f"{args.n_seed}")
    wandb_logger = WandbLogger(name=f"{args.data_space}-{args.n_seed}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_checkpoint = ModelCheckpoint(monitor='val_auc_mean', dirpath=model_dir, filename="{epoch:02d}-{val_auc_mean:.2f}",save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor='val_auc_mean', patience=5, mode='max')

    trainer: pl.Trainer = pl.Trainer(
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=args.n_epochs,
        logger=[wandb_logger, csv_logger],
        logger=[csv_logger],
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
        auto_lr_find=True,
    )
    # Runs a learning rate finder algorithm when calling trainer.tune() to find optimate lr 
    trainer.tune(model, datamodule)
    print("In train_model and {}".format(str(device).startswith("cuda")))
    trainer: pl.Trainer = pl.Trainer(
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=args.n_epochs,
        logger=[wandb_logger, csv_logger],
        logger=[csv_logger],
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
    )
    trainer.fit(model, datamodule)

    return model


# In[ ]:


# test the model 

def test_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:
    
    model_dir = str(args.model_dir) + '/' + args.data_space + '/'  + str(args.n_seed) 
    checkpoint_filename = os.listdir(model_dir)[0]
    print("Checkpoint file: ", model_dir, checkpoint_filename)
    log_dir = (
    Path(args.log_dir)
    / args.data_type
    / args.data_space
    )

    csv_logger = CSVLogger(save_dir=log_dir, name="Test_Complex_CNN", version=f"{args.n_seed}")

    model = RSS.load_from_checkpoint(model_dir + '/' + checkpoint_filename)
    trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0, logger=csv_logger)
    with torch.inference_mode():
        model.eval()
        M_val = trainer.validate(model, datamodule.val_dataloader())  
        M = trainer.test(model, datamodule.test_dataloader())


# In[ ]:



def run_experiment(args):
  
  print(args, flush=True)
  if torch.cuda.is_available():
      print("Found CUDA device, running job on GPU.")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  datamodule = get_data(args)
  data_subset = 
  model = get_model(args, device)
  if args.mode == "train":
      model = train_model(args=args, model=model, datamodule=datamodule, device=device,)
  else:
      datamodule.setup()
      test_model(args=args, model=model, datamodule=datamodule, device=device,)


def main(sweep_step=None):
  args = get_args()
  run_experiment(args)


if __name__ == "__main__":
   fire.Fire(main)

  

