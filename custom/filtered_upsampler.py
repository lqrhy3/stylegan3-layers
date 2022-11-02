import numpy as np
from torch import nn
from torch import Tensor


from torch_utils.ops.upfirdn2d import upsample2d
from .utils import design_lowpass_filter
from .constants import *


class FilteredUpsampler(nn.Module):
    def __init__(
            self,
            sampling_rate: int,
            cutoff: int,
            half_width: int,
            up_factor: int
    ):
        super(FilteredUpsampler, self).__init__()

        self.up_factor = up_factor

        tmp_sampling_rate = sampling_rate * LRELU_UPSAMPLE

        self.tmp_resize_factor = int(np.rint(tmp_sampling_rate / sampling_rate))
        num_taps = FILTER_SIZE * self.tmp_resize_factor
        self.register_buffer('filter', design_lowpass_filter(
            numtaps=num_taps, cutoff=cutoff, width=half_width * 2, fs=tmp_sampling_rate))

    def forward(self, x: Tensor):
        x = upsample2d(
            x=x,
            f=self.filter,
            up=self.up_factor,
            padding=0,  # extra padding
            flip_filter=False,
            gain=1
        )
        return x
