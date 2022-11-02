import torch
from custom.filtered_upsampler import FilteredUpsampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
BATCH_SIZE = 2
NUM_CHANNELS = 3
HEIGHT = 48
WIDTH = 48
SAMPLING_RATE = 48
CUTOFF = 20
HALF_WIDTH = 4
UP_FACTOR = 2


def test_shapes():
    x = torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
    x = x.to(DEVICE)

    filtered_upsampler = FilteredUpsampler(
        sampling_rate=SAMPLING_RATE,
        cutoff=CUTOFF,
        half_width=HALF_WIDTH,
        up_factor=UP_FACTOR
    )

    received_shape = filtered_upsampler(x).shape

    expected_shape = (BATCH_SIZE, NUM_CHANNELS, HEIGHT * 2, WIDTH * 2)

    assert received_shape == expected_shape
