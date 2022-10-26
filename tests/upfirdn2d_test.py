import torch
from torch.nn import functional as F
from torch_utils.ops.upfirdn2d import upfirdn2d

BATCH_SIZE = 2
NUM_CHANNELS = 3
HEIGHT = 48
WIDTH = 48
DTYPE = torch.float32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_updn_identity():
    x = torch.rand((BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH), dtype=torch.float32)
    x = x.to(DEVICE)

    received = upfirdn2d(
        x=x,
        f=None,
        up=2,
        down=2,
        padding=0,
        flip_filter=False,
        gain=1.,
        impl='cuda'
    )

    expected = x

    assert torch.allclose(received, expected)
