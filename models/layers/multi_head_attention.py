from torch import nn as nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split(self, tensor):

        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()

        d_model = head * d_tensor

        tensor = tensor.view(batch_size, length, d_model)   ##concat d_tensor to d_model
        
        return tensor

    def forward(self, q, k, v, mask=None):
        q,k,v = self.w_q(q), self.w_k(k), self.w_v(v)
        q,k,v = self.split(q), self.split(k), self.split(v)

        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out



