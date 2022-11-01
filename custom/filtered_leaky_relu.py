import numpy as np
import torch
from torch import nn
from torch import Tensor

from torch_utils.ops.filtered_lrelu import filtered_lrelu
from training.networks_stylegan3 import SynthesisLayer


design_lowpass_filter = SynthesisLayer.design_lowpass_filter


class FilteredLeakyReLU(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            in_sampling_rate: int,
            out_sampling_rate: int,
            in_cutoff: int,
            out_cutoff: int,
            in_half_width: int,
            out_half_width: int,
            conv_kernel: int
    ):
        super(FilteredLeakyReLU, self).__init__()

        lrelu_upsampling = 2
        filter_size = 6
        down_radial = False

        tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * lrelu_upsampling
        in_size = np.broadcast_to(np.asarray(in_size), [2])
        out_size = np.broadcast_to(np.asarray(out_size), [2])

        # Design upsampling filter.
        self.up_factor = int(np.rint(tmp_sampling_rate / in_sampling_rate))
        up_taps = filter_size * self.up_factor if self.up_factor > 1 else 1
        self.register_buffer('up_filter', design_lowpass_filter(
            numtaps=up_taps, cutoff=in_cutoff, width=in_half_width * 2, fs=tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(tmp_sampling_rate / out_sampling_rate))
        down_taps = filter_size * self.down_factor if self.down_factor > 1 else 1
        self.register_buffer('down_filter', design_lowpass_filter(
            numtaps=down_taps, cutoff=out_cutoff, width=out_half_width * 2, fs=tmp_sampling_rate,
            radial=down_radial))

        # Compute padding.
        pad_total = (out_size - 1) * self.down_factor + 1  # Desired output size before downsampling.
        pad_total -= (in_size + conv_kernel - 1) * self.up_factor  # Input size after upsampling.
        pad_total += up_taps + down_taps - 2  # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2  # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x: Tensor):
        x = filtered_lrelu(
            x=x,
            fu=self.up_filter,
            fd=self.down_filter,
            b=None,
            up=self.up_factor,
            down=self.down_factor,
            padding=self.padding,
            gain=1.,
            slope=1e-2,
            clamp=None
        )

        return x


if __name__ == '__main__':
    flrelu = FilteredLeakyReLU(
        in_size=28,
        out_size=28,
        in_sampling_rate=28,
        out_sampling_rate=28,
        in_cutoff=10,
        out_cutoff=10,
        in_half_width=4,
        out_half_width=4,
        conv_kernel=1,
    )

    x = torch.rand((2, 3, 28, 28), dtype=torch.float32)
    out = flrelu(x)
    print(out.shape)
