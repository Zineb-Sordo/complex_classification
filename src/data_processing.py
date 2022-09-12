import os
import numpy as np
import h5py
import pandas as pd
from collections import namedtuple
import math
from tqdm import tqdm

from typing import List, Optional, Sequence, Tuple, Union
from joblib import dump, load
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler

from typing import Dict, Optional, Tuple
import pytorch_lightning as pl
import torch
from torchvision import transforms

from pathlib import Path

from fastmri.fftc import ifft2c_new
import fastmri
from fastmri.data import transforms as T


class MultiDataset(Dataset):
    def __init__(self, split_csv_file: str, mode: str, dev_mode: bool = False):
        super().__init__()
        # read csv file with filenames
        self.split_csv_file = Path(split_csv_file)
        assert self.split_csv_file.is_file()
        self.metadata = pd.read_csv(self.split_csv_file)
        if dev_mode:
            self.metadata = self.metadata.iloc[:50]

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


# Defining the Sample namedtuple at module level
fields = [
    "volume_id",
    "slice_id",
    "sc_kspace",
    "sc_kspace_scaled",
    "mc_kspace",
    "recon_esc",
    "recon_rss",
    "label",
    "data_split",
    "dataset",
    "location",
    "max_value"
]
Sample = namedtuple("Sample", fields, defaults=(math.nan,) * len(fields))


def center_crop(data, shape: Tuple[int, int]):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


class KneeDataset(MultiDataset):
    def __init__(
            self,
            split_csv_file: str,
            mode: str,
            label_type: str,
            data_space: str,
            coil_type='sc',
            image_only: bool = False,
            image_shape=[320,320],
    ):
        super().__init__(split_csv_file=split_csv_file, mode=mode)

        self.coil_type = coil_type
        assert self.coil_type in {"sc", "mc", "sc_scaled"}
        self.label_type = label_type
        self.data_space = data_space
        self.mode = mode
        self.image_shape = image_shape

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
        if self.coil_type == "sc":
            kspace_key = "sc_kspace"
        elif self.coil_type == "sc_scaled":
            kspace_key = "sc_kspace_scaled"
        else:
            kspace_key = "mc_kspace"
        target_key = "recon_esc" if self.coil_type == "sc" else "recon_rss"

        with h5py.File(loc) as f:
            kspace_data = f[kspace_key][:]
            target_data = f[target_key][:]

            if kspace_key == "sc_kspace":
                image_data = fastmri.ifft2c(T.to_tensor(kspace_data))
                kspace_data = torch.complex(image_data[:, :, 0], image_data[:, :, 1])
                kspace_data = center_crop(kspace_data, self.image_shape)
                # image_data = torch.view_as_real(torch.from_numpy(kspace_data).float())
                # image_data = ifft2c_new(image_data)
                # kspace_data = torch.view_as_complex(image_data)

            parameters = {
                kspace_key: kspace_data,
                target_key: target_data,
                "volume_id": info.volume_id,
                "slice_id": info.slice_id,
                "data_split": info.data_split,
                "dataset": info.dataset,
                "location": info.location,
            }
            # print("in parameters the kspace dtype is {}".format(parameters['sc_kspace'].dtype))
            if self.label_type == "knee_multilabel":
                parameters["label"] = self.parse_multilabel(info.labels)
            elif self.label_type == "knee":
                parameters["label"] = self.parse_label(info.labels)
            else:
                raise NotImplementedError(
                    f"Label type {self.label_type} not implemented"
                )



        sample = Sample(**parameters)
        # print("the sample shape is {}".format(sample.sc_kspace.shape))
        return sample


def get_sampler_weights(dataset, save_filename="./sampler_knee_tr.p"):
    Y_tr = []

    for i in tqdm(range(len(dataset))):
        label = dataset[i].label.sum().item()
        Y_tr.append(label)

    Y_tr = np.array(Y_tr).astype(int)

    class_sample_count = np.array(
        [len(np.where(Y_tr == t)[0]) for t in np.unique(Y_tr)]
    )

    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in Y_tr])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dump(sampler, save_filename)


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
        num_workers: int = 4,
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
        self.num_workers = num_workers

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

        if self.sampler_filename is None:
            print("Creating sampler weights...")
            self.sampler_filename = "sampler_knee_tr.p"
            get_sampler_weights(self.train_dataset, self.sampler_filename)
        elif not os.path.exists(self.sampler_filename):
            print(self.sampler_filename, os.getcwd())
            raise ValueError("Weighted sampler does not exist")
        assert Path(self.sampler_filename).is_file()
        # load the sampler
        self.train_sampler = load(self.sampler_filename)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          sampler=self.train_sampler,
                          num_workers=self.num_workers,)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

