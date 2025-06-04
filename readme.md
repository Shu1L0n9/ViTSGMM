# ViTSGMM: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels

Official implementation of ViTSGMM: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitsgmm-a-robust-semi-supervised-image-1/semi-supervised-image-classification-on-cifar-7)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-7?p=vitsgmm-a-robust-semi-supervised-image-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitsgmm-a-robust-semi-supervised-image-1/semi-supervised-image-classification-on-stl-3)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-stl-3?p=vitsgmm-a-robust-semi-supervised-image-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/vitsgmm-a-robust-semi-supervised-image-1/semi-supervised-image-classification-on-cifar-8)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-8?p=vitsgmm-a-robust-semi-supervised-image-1)

## Citation

If you use this code or ideas from our work, please cite:

```bibtex
@misc{yann2025vitsgmm,
      title={ViTSGMM: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels},
      author={Rui Yann and Xianglei Xing},
      year={2025},
      eprint={2506.03582},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.03582}
}
```

## Todo

- [x] translate to English
- [ ] add stl10 duplication code
- [ ] add pseudo-label into pipeline

## Project Structure

```
ViTSGMM/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── LICENSE                      # Apache LICENSE
├── src/                         # Source code directory
│   ├── __init__.py
│   ├── config/                  # Configuration files
│   │   ├── __init__.py
│   │   └── config.py           # Project configuration
│   ├── data/                    # Data processing module
│   │   ├── __init__.py
│   │   └── data_loader.py      # Data loader
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   └── sgmm_core.py        # SGMM core implementation
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── utils.py            # General utilities
├── scripts/                     # Execution scripts
│   ├── extract_features.py     # Feature extraction script
│   └── train_sgmm.py          # Training script
├── data/                        # Data directory
│   ├── cifar-10-batches-py/    # CIFAR-10 data
│   ├── cifar-100-python/       # CIFAR-100 data
│   └── vit/                    # ViT features
│       ├── cifar10/
│       ├── cifar100/
│       └── stl10/
└── experiment.ipynb        # Experiment notebook
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2. Jupyter experiments

```bash
jupyter lab experiment.ipynb
```

## Main Features

- **Pretrained ViT feature extraction**: Extract high-quality image features using DINOv2 pretrained Vision Transformer
- **Semi-supervised learning**: Combine a small amount of labeled data with a large amount of unlabeled data for learning
- **Gaussian Mixture Model**: Model the feature space using GMM
- **Multi-dataset support**: Supports CIFAR-10, CIFAR-100, and STL-10 datasets
- **Configurable design**: Easy to adjust parameters and extend

## Configuration

The main configuration file is located at `src/config/config.py`, including:

- **Data config**: Dataset type, path, batch size, etc.
- **Model config**: Number of PCA components, number of GMM components, convergence threshold, etc.
- **ViT config**: Pretrained model selection, device configuration, etc.

## Dependencies

Main dependencies include:

- PyTorch
- timm (PyTorch Image Models)
- scikit-learn
- numpy
- tqdm

See `requirements.txt` for details.

## License

Apache License
