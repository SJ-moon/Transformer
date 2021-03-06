import torch
from torch import nn as nn

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) ## Parameter 와 nn.tensor의 차이점: parameter는 paramter()로 검색이 가능
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)

        out = (x-mean)/(std + self.eps)
        out = self.gamma * out + self.beta
        return out