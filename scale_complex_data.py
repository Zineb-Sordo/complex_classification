import torch
import numpy as np
import pandas as pd
import h5py
import tqdm
from typing import Tuple

import fastmri
from fastmri.data import transforms as T


def center_crop(data, shape: Tuple[int, int]):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


if __name__ == '__main__':

    csv_file = "./metadata_knee.csv"
    df = pd.read_csv(csv_file)
    df_train = df[df.data_split == "train_class"]
    df_val_test = df[(df.data_split == "val_class") | (df.data_split == "test_class")]
    print("The number of training examples is {} and val/test examples is {}".format(len(df_train),len(df_val_test)))

    # Get training h5 file and add them all to list to concat and get mean and cov (after IFFT and cropping the data)
    list_train_paths = list(df_train.location)
    list_train_kspace, list_h5_files_train = [], []
    for file_path in tqdm.tqdm(list_train_paths):
        file = h5py.File(file_path, 'r+')
        # list_h5_files_train.append(file)
        list_train_kspace.append(file['sc_kspace'][()])
        file.close()

    l_out_train_fft = [0] * len(list_train_kspace)
    for i in range(len(list_train_kspace)):
        l_out_train_fft[i] = fastmri.ifft2c(T.to_tensor(list_train_kspace[i]).float())
        l_out_train_fft[i] = torch.complex(l_out_train_fft[i][:, :, 0], l_out_train_fft[i][:, :, 1]).type(torch.complex64)
        l_out_train_fft[i] = center_crop(l_out_train_fft[i], [320, 320])

    vol_train_out = torch.concat(l_out_train_fft)
    # get mean
    mr, mi = vol_train_out.real.mean().type(torch.complex64), vol_train_out.imag.mean().type(torch.complex64)
    mean = mr + 1j * mi

    # get covariance
    eps = 0
    n = vol_train_out.numel() / vol_train_out.size(1)
    Crr = 1. / n * vol_train_out.real.pow(2).sum() + eps
    Cii = 1. / n * vol_train_out.imag.pow(2).sum() + eps
    Cri = (vol_train_out.real.mul(vol_train_out.imag)).mean()

    # calculate the inverse square root the covariance matrix
    det = Crr * Cii - Cri.pow(2)
    s = torch.sqrt(det)
    t = torch.sqrt(Cii + Crr + 2 * s)
    inverse_st = 1.0 / (s * t)
    Rrr = (Cii + s) * inverse_st
    Rii = (Crr + s) * inverse_st
    Rri = -Cri * inverse_st

    print("Now scaling the training examples")
    # Scale the train data slices
    for i in tqdm.tqdm(range(len(l_out_train_fft))):
        slice_out = l_out_train_fft[i] - mean
        l_out_train_fft[i] = (Rrr[None, None] * slice_out.real + Rri[None, None] * slice_out.imag).type(torch.complex64) \
                       + 1j * (Rii[None, None] * slice_out.imag + Rri[None, None] * slice_out.real).type(torch.complex64)

    for i in tqdm.tqdm(range(len(list_train_paths))):
        file_train = h5py.File(list_train_paths[i], 'r+')
        file_train.create_dataset("sc_kspace_scaled", data=l_out_train_fft[i])
        file_train.close()

    # for i in range(len(list_h5_files_train)):
    #     file_train = list_h5_files_train[i]
    #     file_train.create_dataset("sc_kspace_scaled", data=l_out_train_fft[i])
    #     file_train.close()

    # Do the same with the val and test dataframe using the mean and cov of the training set
    list_val_paths = list(df_val_test.location)
    list_val_kspace, list_h5_files_val_test = [], []
    for file_path in tqdm.tqdm(list_val_paths):
        file = h5py.File(file_path, 'r+')
        # list_h5_files_val_test.append(file)
        list_val_kspace.append(file['sc_kspace'][()])

    l_out_val_fft = [0] * len(list_val_kspace)
    for i in range(len(l_out_val_fft)):
        l_out_val_fft[i] = fastmri.ifft2c(T.to_tensor(list_val_kspace[i]).float())
        l_out_val_fft[i] = torch.complex(l_out_val_fft[i][:, :, 0], l_out_val_fft[i][:, :, 1]).type(torch.complex64)
        l_out_val_fft[i] = center_crop(l_out_val_fft[i], [320, 320])

    # Scaling of the val and test data slices with the train mean and cov previously calculated
    for i in tqdm.tqdm(range(len(l_out_val_fft))):
        slice_out = l_out_val_fft[i] - mean
        l_out_val_fft[i] = (Rrr[None, None] * slice_out.real + Rri[None, None] * slice_out.imag).type(torch.complex64)\
                            + 1j * (Rii[None, None] * slice_out.imag + Rri[None, None] * slice_out.real).type(torch.complex64)

    # for i in range(len(list_h5_files_val_test)):
    #     file_val_test = list_h5_files_val_test[i]
    #     file_val_test.create_dataset("sc_kspace_scaled", data=l_out_val_fft[i])
    #     file_val_test.close()

    for i in tqdm.tqdm(range(len(list_val_paths))):
        file_val_test = h5py.File(list_val_paths[i], 'r+')
        file_val_test.create_dataset("sc_kspace_scaled", data=l_out_val_fft[i])
        file_val_test.close()


