import torch
from custom.filtered_downsampler import FilteredDownsampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32
BATCH_SIZE = 2
NUM_CHANNELS = 3
HEIGHT = 48
WIDTH = 48
SAMPLING_RATE = 48
CUTOFF = 10
HALF_WIDTH = 2
DOWN_FACTOR = 2


def test_shapes():
    x = torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
    x = x.to(DEVICE)

    filtered_downsampler = FilteredDownsampler(
        sampling_rate=SAMPLING_RATE,
        cutoff=CUTOFF,
        half_width=HALF_WIDTH,
        down_factor=DOWN_FACTOR
    )

    received_shape = filtered_downsampler(x).shape

    expected_shape = (BATCH_SIZE, NUM_CHANNELS, HEIGHT // 2, WIDTH // 2)

    assert received_shape == expected_shape
