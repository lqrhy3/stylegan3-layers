import torch
from torch.nn import functional as F
from torch_utils.ops.filtered_lrelu import filtered_lrelu
from custom.filtered_leaky_relu import FilteredLeakyReLU

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
BATCH_SIZE = 2
NUM_CHANNELS = 3
HEIGHT = 48
WIDTH = 48
SAMPLING_RATE = 48
CUTOFF = 20
HALF_WIDTH = 4


def test_shapes():
    x = torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
    x = x.to(DEVICE)

    filtered_leaky_relu = FilteredLeakyReLU(
        sampling_rate=SAMPLING_RATE,
        cutoff=CUTOFF,
        half_width=HALF_WIDTH,
        negative_slope=1e-2
    )
    filtered_leaky_relu.to(DEVICE)

    received_shape = filtered_leaky_relu(x).shape

    expected_shape = (BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)

    assert received_shape == expected_shape
