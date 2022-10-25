import torch
from torch_utils.ops.filtered_lrelu import filtered_lrelu

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand((2, 3, 48, 48), dtype=torch.float32)
    x = x.to(device)
    
    y = filtered_lrelu(x)

