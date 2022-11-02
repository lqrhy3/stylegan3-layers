import torch
from torch.nn import functional as F
from torch_utils.ops.filtered_lrelu import filtered_lrelu

BATCH_SIZE = 2
NUM_CHANNELS = 3
HEIGHT = 48
WIDTH = 48
DTYPE = torch.float32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_non_filtered_lrelu():
    x = torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
    x = x.to(DEVICE)

    received = filtered_lrelu(
        x=x,
        fu=None,
        fd=None,
        b=None,
        up=1,
        down=1,
        padding=0,
        gain=1.,
        slope=0.2,
        clamp=None,
        flip_filter=False,
        impl='cuda'
    )

    expected = F.leaky_relu(
        input=x,
        negative_slope=0.2,
        inplace=False
    )

    assert torch.allclose(received, expected)
