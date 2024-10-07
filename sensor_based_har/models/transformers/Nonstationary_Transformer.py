import torch
import torch.nn as nn
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from .layers.SelfAttention_Family import DSAttention, AttentionLayer
from .layers.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(
            self, seq_len, output_attention=False, enc_in=1,
            d_model=512, embed='timeF', freq='h', dropout=0.1,
            d_ff=2048, n_heads=8, e_layers=3, activation='gelu',
            num_class=10, p_hidden_dims=[64], p_hidden_layers=1,
            seq_len_last=False,
        ):
        """
        Parameters
        ----------
        seq_len: int
            sequence length
        output_attention: bool
            whether to output attention
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
        p_hidden_dims: list
            hidden dimensions for the projector
        p_hidden_layers: int
            number of layers for the projector
        seq_len_last: bool
            whether to use the last sequence length. seq_len_last is True -> (batch_size, enc_in, seq_len)
        """
        super(Model, self).__init__()
        task_name = 'classification'
        self.task_name = task_name
        self.seq_len = seq_len
        self.output_attention = output_attention
        self.enc_in = enc_in
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout_rate = dropout
        # self.factor = factor
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers
        self.num_class = num_class
        self.p_hidden_dims = p_hidden_dims
        self.p_hidden_layers = p_hidden_layers
        self.seq_len_last = seq_len_last
        factor = 1  # NOTE: this is not used in DSAttention

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
                        DSAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model,
                        n_heads
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

        self.tau_learner = Projector(
            enc_in=enc_in,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=1
        )
        self.delta_learner = Projector(
            enc_in=enc_in,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=seq_len
        )

    def classification(self, x_enc, x_mark_enc):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc=None):
        # NOTE: x_enc.shape = (batch_size, seq_length, enc_in)
        assert self.task_name == 'classification'
        if self.seq_len_last:
            x_enc = x_enc.transpose(1, 2)
        if x_mark_enc is None:
            x_mark_enc = torch.ones(x_enc.shape[0], x_enc.shape[1]).to(x_enc.device)
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, L, D]
