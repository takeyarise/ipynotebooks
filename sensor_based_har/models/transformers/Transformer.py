# from: https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(
            self, output_attention=False, enc_in=1,
            d_model=512, embed='timeF', freq='h', dropout=0.1,
            d_ff=2048, n_heads=8, e_layers=3, activation='gelu',
            num_class=2, seq_len=24, seq_len_last=False,
        ):
        """
        Vanilla Transformer

        Parameters
        ----------
        output_attention: bool
            whether to output attention, 対応していない
        enc_in: int
            input feature dimension
        d_model: int
            model dimension
        embed: str
            embedding type
        freq: str
            frequency
        dropout: float
            dropout rate
        d_ff: int
            feedforward dimension
        n_heads: int
            number of heads
        e_layers: int
            number of layers
        activation: str
            activation function
        num_class: int
            number of classes
        seq_len: int
            sequence length
        seq_len_last: bool
            whether to use the last sequence length. seq_len_last is True -> (batch_size, enc_in, seq_len)
        """
        super(Model, self).__init__()
        task_name = 'classification'
        self.task_name = task_name
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        # self.factor = factor
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.activation = activation
        self.num_class = num_class
        self.seq_len = seq_len
        self.seq_len_last = seq_len_last
        factor = 1  # NOTE: this is not used in FullAttention

        # Embedding
        self.enc_embedding = DataEmbedding(
            enc_in,
            d_model,
            embed,
            freq,
            dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * seq_len, num_class)

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None):
        # NOTE: x_enc.shape = (batch_size, seq_length, enc_in)
        assert self.task_name == 'classification'
        if self.seq_len_last:
            x_enc = x_enc.transpose(1, 2)
        if x_mark_enc is None:
            x_mark_enc = torch.ones(x_enc.shape[0], x_enc.shape[1]).to(x_enc.device)
        return self.classification(x_enc, x_mark_enc)
