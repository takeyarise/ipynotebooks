# README

from: https://github.com/thuml/Time-Series-Library

デバッグしていない:
- TimesNet
- DLinear


```python
net = Nonstationary_Transformer.Model(
    output_attention=False,  # 対応していない
    enc_in=3,  # channel size
    d_model=256,  # hidden dim
    embed='timeF',  # 'timeF' 固定でいいと思う
    freq='b',  # 'b' 固定でいいと思われる
    dropout=0.1,
    d_ff=512,  # hidden dim at feedforward
    n_heads=8,
    e_layers=3,
    activation='gelu',
    num_class=6,
    seq_len=512,  # window size
    p_hidden_dims=[64],  # 不明
    p_hidden_layers=1,  # 不明
    seq_len_last=True,  # 入力データの shape が (batch size, channels, window size) の場合は True にする
)
```

```python
net = Transformer.Model(
    output_attention=False,  # 対応していない
    enc_in=3,  # channel size
    d_model=256,  # hidden dim
    embed='timeF',  # 'timeF' 固定でいいと思う
    freq='b',  # 'b' 固定でいいと思われる
    dropout=0.1,
    d_ff=512,  # feedforward の hidden dim
    n_heads=8,
    e_layers=3,  # N = 6 (orig paper)
    activation='gelu',
    num_class=6,
    seq_len=512,  # window size
    seq_len_last=True,  # (batch size, channels, window size) の場合は True, (batch size, window size, channels) の場合は False
)
```
