
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
#get_ipython().run_line_magic('matplotlib', 'inline')

import torch.nn as nn
from torch import optim
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
from typing import Dict
from sklearn import metrics
import torch
from torchmetrics import functional
import numpy as np
from complex_cnn_model import complex_resnet18_knee


def compute_auc(preds, labels):
    return functional.auroc(preds=preds, target=labels, num_classes=2)


def compute_accuracy(preds, labels):
    return functional.accuracy(preds=preds, target=labels)


def get_operating_point(preds, labels, operating_point=None, threshold=0.1):
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

    # print(f"b acc = {balanced_acc}, specificity = {specificity}, sensitivity = {sensitivity}")
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

    print("operating point {}".format(operating_point))

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


def get_model(
    data_type: str,
    model_type: str,
    drop_prob: float,
    data_space: str,
    image_shape: Tuple[int, int],
    sequences: Optional[Tuple[str, str, str]] = None,
    return_features=False,
    num_labels=4
) -> pl.LightningModule:
    if data_type == "knee":
        if model_type == "complex_preact_resnet18":
            return complex_resnet18_knee(image_shape=image_shape, drop_prob=drop_prob, data_space=data_space, return_features=return_features)
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
        print("batch.sc_kspace shape is {}".format(kspace.shape))
        kspace = kspace.type(torch.complex32)
        print("dtype of the kspace is {}".format(kspace.dtype))
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
            
            self.log(label_names[i], acc, prog_bar=True, sync_dist=True)
            
        return loss

    def training_step(self, batch, batch_idx):
        labels = batch.label.long()
        # get predictions
        print("in training_step the batch shape and batch_idx are {} and {}".format(batch.sc_kspace.shape, batch_idx))
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
        print("in optimizer")
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
        )

        return [optimizer], [scheduler]