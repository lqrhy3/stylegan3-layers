import numpy as np
import torch
from torch import nn
from torch import Tensor

from torch_utils.ops.filtered_lrelu import filtered_lrelu
from .utils import design_lowpass_filter
from .constants import *


class FilteredLeakyReLU(nn.Module):
    def __init__(
            self,
            sampling_rate: int,
            cutoff: int,
            half_width: int,
            negative_slope: float = 1e-2
    ):
        super(FilteredLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

        tmp_sampling_rate = sampling_rate * LRELU_UPSAMPLE

        self.tmp_resize_factor = int(np.rint(tmp_sampling_rate / sampling_rate))
        num_taps = FILTER_SIZE * self.tmp_resize_factor
        self.register_buffer('filter', design_lowpass_filter(
            numtaps=num_taps, cutoff=cutoff, width=half_width * 2, fs=tmp_sampling_rate))

        # Compute padding required to return same sized tensor
        pad_total = np.array([-1, -1])  # TODO: Derived from SynthesisLayer padding calculation,
                                        #       but what sense does it have?
        pad_total += (num_taps - 1) * 2  # Size reduction caused by the filters.
        pad_lo = (pad_total + self.tmp_resize_factor) // 2  # Shift sample locations according to the symmetric interpretation (Appendix C.3).
                                                            # TODO: what sense does it have?
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x: Tensor):
        assert x.ndim == 4

        x = filtered_lrelu(
            x=x,
            fu=self.filter,
            fd=self.filter,
            b=None,
            up=self.tmp_resize_factor,
            down=self.tmp_resize_factor,
            padding=self.padding,
            gain=1.,
            slope=self.negative_slope,
            clamp=None
        )

        return x
