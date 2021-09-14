import math
from torch import nn

class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention,self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size() ## Ref. Tensor.size()
        k_t = k.view(batch_size, head, d_tensor, length) ## transpose
        score = (q@k_t) / math.sqrt(d_tensor) ## @ means matmul

        if mask is not None:
            score = score.masked_fill(mask==0, -e)      #make infinite where mask is 0

        score = self.softmax(score)

        v= score@v
        
        return v, score


