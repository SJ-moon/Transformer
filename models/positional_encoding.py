import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):

        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False ## don't need to compute grad

        pos = torch.arrange(0, max_len)
        pos = pos.float().unsqueeze(dim=1) ## 1D to 2D

        i_2 = torch.arange(0, d_model, step=2) ## 2*i
        self.encoding[:,0::2] = torch.sin(pos/(10000 ** (i_2 / d_model)))
        self.encoding[:,1::2] = torch.cos(pos / (10000 ** (i_2/ d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]