from torch import nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_normalization import LayerNorm
from models.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden. n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model = d_model, n_head = n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        _x = x  ##save x
        x = self.attention(q=x, k=x, v =x, mask=s_mask)

        x = self.norm1(x+ _x)
        x = self.dropout1(x)

        _x = x      ##save x
        x = self.ffn(x)

        x  = self.norm2(x+ _x)
        x = self.dropout2(x)
        return x


class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Encoder,self).__init__()
        self.emb = Embedding ## not finished...

        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model, ffn_hidden = ffn_hidden, n_head = n_head, drop_prob = drop_prob) for _ in range(n_layers)])        ## 좋은 forward 간략화 사용법!

    def forward(self, x, s_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, s_mask)

        return x