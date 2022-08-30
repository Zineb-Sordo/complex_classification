

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


def center_crop(data, shape: Tuple[int, int]):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


class ComplexPreActBlock(pl.LightningModule):
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


class ComplexPreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(ComplexPreActBottleneck, self).__init__()
        self.bn1 = ComplexBatchNorm2d(in_planes)
        self.conv1 = ComplexConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = ComplexBatchNorm2d(planes)
        self.conv2 = ComplexConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = ComplexBatchNorm2d(planes)
        self.conv3 = ComplexConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
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
        out = complex_relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def complex_dropout2d(input, p=0.5, training=True):
    mask = torch.ones(*input.shape, dtype=torch.float32, device=torch.device('cuda'))
    #mask = torch.ones(*input.shape, dtype = torch.float32) #when using CPU only
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
            num_blocks,
            image_shape,
            data_space,
            num_classes=4,
            drop_prob=0.5,
            block=[2, 2, 2, 2],
            return_features=False,

    ):
        super(ComplexPreActResNetFFTKnee, self).__init__()
        self.in_planes = 64

        self.conv_comp = ComplexConv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1_p = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.dropout = ComplexDropout2d(p=drop_prob)
        self.image_shape = image_shape
        self.data_space = data_space

        in_dim = 512 * block.expansion * 100

        self.linear_mtear = nn.Linear(8, num_classes)
        self.linear_acl = nn.Linear(8, num_classes)
        self.linear_abnormal = nn.Linear(8, num_classes)
        self.linear_cartilage = nn.Linear(8, num_classes)

        self.Clinear_mtear = ComplexLinear(in_dim, num_classes)
        self.Clinear_acl = ComplexLinear(in_dim, num_classes)
        self.Clinear_abnormal = ComplexLinear(in_dim, num_classes)
        self.Clinear_cartilage = ComplexLinear(in_dim, num_classes)

        self.complexLinear = ComplexLinear(in_dim, num_classes)
        # self.bn1d = ComplexBatchNorm1d(out_dim, track_running_stats = False)
        # self.complexLinear2 = ComplexLinear(out_dim, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, kspace):
        # print("the kspace shape is {} and dtype is {}".format(kspace.shape, kspace.dtype)) # torch.size([8, 1, 640, 400])
        if self.data_space == 'complex_input':
            out = torch.complex(kspace.real, kspace.imag).cuda().type(torch.complex64)
            #out = torch.complex(kspace.real, kspace.imag).type(torch.complex64)
            out = center_crop(out, self.image_shape)
            out = self.conv_comp(out)
        out = self.dropout(out)

        layer_1_out = self.layer1(out)
        layer_2_out = self.layer2(layer_1_out)
        layer_3_out = self.layer3(layer_2_out)
        layer_4_out = self.layer4(layer_3_out)
        out = complex_avg_pool2d(layer_4_out, 4)
        out = out.view(out.size(0), -1)

        #out = self.dropout(out)
        #out = complex_relu(out)

        out_mtear = self.Clinear_mtear(out)
        out_acl = self.Clinear_acl(out)
        out_cartilage = self.Clinear_cartilage(out)
        out_abnormal = self.Clinear_abnormal(out)

        # First approach: output is magnitude
        out_mtear = out_mtear.abs()
        out_acl = out_acl.abs()
        out_cartilage = out_cartilage.abs()
        out_abnormal = out_abnormal.abs()

        # Third approach is use a convolution of the magnitude and phase channels
        #print("outputs = {}, {}, {}, {}".format(out_abnormal, out_mtear, out_acl, out_cartilage))
        return out_abnormal, out_mtear, out_acl, out_cartilage


# def complex_resnet18_knee(image_shape, data_space, drop_prob=0.3, return_features=False):
#     return ComplexPreActResNetFFTKnee(
#         ComplexPreActBlock,
#         [2, 2, 2, 2],
#         drop_prob=drop_prob,
#         image_shape=image_shape,
#         data_space=data_space,
#         return_features=return_features
#     )


class RSS(pl.LightningModule):
    def __init__(self,
                 model_type: str,
                 data_type: str,
                 drop_prob: float,
                 kspace_shape: Tuple[int, int],
                 image_shape: Tuple[int, int],
                 device: torch.device,
                 data_space: str,
                 # label_names: str,
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
        # self.label_names = label_names
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


    def loss_fn(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(preds, labels)

    def compute_loss_and_metrics(self, preds, labels, label_names):

        assert len(label_names) == self.num_labels

        pred_out, label_out = [], [] # To store preds and labels for each label
        acc_per_label = []

        loss = None
        # print("num labels:", self.num_labels)
        for i in range(0, self.num_labels):

            curr_loss = self.loss_fn(preds=preds[i], labels=labels[: ,i])
            if loss is None:
                loss = curr_loss
            else:
                loss += curr_loss

            acc = compute_accuracy(preds[i], labels[:, i])
            acc_per_label.append(acc)

            self.log(label_names[i], acc, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        labels = batch.label.long()
        # get predictions
        kspace = batch.sc_kspace
        kspace = kspace.to(device=self.device).type(torch.complex64)
        kspace = kspace.unsqueeze(1)

        preds = self.forward(kspace)

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
        # print("in validation step, kspace shape {}".format(batch.sc_kspace.shape))

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
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

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
                self.log(f"val_auc_{key}", key_score, prog_bar=True, sync_dist=True)
                self.log(
                    f"val_bac_{key}",
                    eval_metrics[key]["balanced_accuracy"],
                    prog_bar=True,
                    sync_dist = True
                )
                avg_auc += key_score / len(keys)

                self.val_operating_point[key] = eval_metrics[key]["operating_point"]

            self.log(f"val_auc_mean", avg_auc, prog_bar=True, sync_dist=True)

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
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

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
                        sync_dist=True
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