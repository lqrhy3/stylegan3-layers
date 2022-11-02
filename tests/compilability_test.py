import torch.cuda

from torch_utils.ops import filtered_lrelu
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act


def test_filtered_lrelu_compilability():
    if torch.cuda.is_available():
        assert filtered_lrelu._init()
    else:
        assert True  # no compilation needed if there is no CUDA


def test_upfirdn2d_compilability():
    if torch.cuda.is_available():
        assert upfirdn2d._init()
    else:
        assert True  # no compilation needed if there is no CUDA


def test_bias_act_compilability():
    if torch.cuda.is_available():
        assert bias_act._init()
    else:
        assert True  # no compilation needed if there is no CUDA
