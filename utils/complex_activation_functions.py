import torch
import numpy as np
import torch.nn as nn


def zReLU(input):
    a, b = input.real, input.imag
    mask = ((0 < input.angle()) * (input.angle() < np.pi/2)).float()
    real = a * mask
    imag = b * mask
    result = real.type(torch.complex64) + 1j*imag.type(torch.complex64)
    return result


def modReLU(input, bias):

    a, b = input.real, input.imag
    input_mag = input.abs()
    mask = ((input_mag + bias) >= 0).float() * (1 + bias / input_mag)
    real = mask * a
    imag = mask * b
    result = real.type(torch.complex64) + 1j*imag.type(torch.complex64)
    return result


def cardioid(input):
    phase = input.angle()
    mask = 0.5*(1 + torch.cos(phase)).float()
    a, b = input.real, input.imag
    real = a * mask
    imag = b * mask
    result = real.type(torch.complex64) + 1j * imag.type(torch.complex64)
    return result
