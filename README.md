# ipynotebooks
Jupyter Notebook Collections

実験用

<details>
<summary>memo</summary>

**common.optim 関連**
- `common/optim/asam.py` は元のまま
- `common/optim/ranger21.py` は試していない

</details>

## directory

案:
```
ipynotebooks/
├── common/
│   └── utils.py
├── sensor_based_har/
│   ├── data/
│   ├── weights/
│   ├── models/
│   ├── notebooks/
│   └── scripts/
├── image_recognition/
│   ├── data/
│   ├── weights/
│   ├── models/
│   ├── notebooks/
│   └── scripts/
└── README.md
```

```bash
ln -s /path/to/data data
ln -s /path/to/weights weights
```

## installed

- pytorch, torchvision, torchaudio
- numpy
- pandas
- matplotlib
- seaborn
- jupyterlab
- ipywidgets
- timm
- tqdm
- einops
- rich
- lightning
- python-dotenv
- scikit-learn
- umap-learn
- torchinfo
**pip**
- torcheval
