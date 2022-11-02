# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import scipy.signal
import scipy.optimize
import torch


def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
    assert numtaps >= 1

    # Identity filter.
    if numtaps == 1:
        return None

    # Separable Kaiser low-pass filter.
    if not radial:
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    # Radially symmetric jinc-based filter.
    x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
    r = np.hypot(*np.meshgrid(x, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
    beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
    w = np.kaiser(numtaps, beta)
    f *= np.outer(w, w)
    f /= np.sum(f)
    return torch.as_tensor(f, dtype=torch.float32)