from torch import nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.layer_normalization import LayerNorm
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_dead, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)


    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.self_attention(q = dec, k = dec, v = dec, mask = t_mask)

        x= self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k = enc, v= enc, mask = s_mask)

            x= self.norm2(x+_x)
            x= self.dropout2(x)

        _x = x
        x = self.ffn(x)

        x = self.norm3(x + _x)
        x= self.dropout3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Decoder, self).__init__()
        self.emb = Embedding ## not finished...

        self.layers = nn.ModuleList([DecoderLayer(d_model = d_model, ffn_hidden = ffn_hidden, n_head = n_head, drop_prob = drop_prob) for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)           ## layer.forward
        
        output = self.linear(trg)
        return output