from torch_utils.ops import filtered_lrelu
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act


def test_filtered_lrelu_compilability():
    assert filtered_lrelu._init()


def test_upfirdn2d_compilability():
    assert upfirdn2d._init()


def test_bias_act_compilability():
    assert bias_act._init()

