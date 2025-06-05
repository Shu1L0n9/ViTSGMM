# ViTSGMM: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels

Official implementation of ViTSGMM: A Robust Semi-Supervised Image Recognition Network Using Sparse Labels.

[![arXiv](https://img.shields.io/badge/arXiv-2506.03582-b31b1b.svg)](https://arxiv.org/abs/2506.03582)
[![HuggingFace Dataset](https://img.shields.io/badge/dataset%20on-HuggingFace-blue?logo=huggingface)](https://huggingface.co/datasets/Shu1L0n9/CleanSTL-10)
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

## 🧼 Clean STL-10 Dataset

🎉 We uploaded our cleaned STL-10 dataset to Hugging Face! You can easily load and use it with the 🤗 `datasets` library.

### 🔧 How to Load

```python
from datasets import load_dataset

# Load the cleaned STL-10 dataset
dataset = load_dataset("Shu1L0n9/CleanSTL-10", split="train")
# Other available splits: "test", "unlabeled"
```

## 📘 Basic SGMM Methods

We believe Miller and Uyar’s SGMM method has been underestimated due to the lack of available implementations.  
To support learning and research, we’ve open-sourced our SGMM implementations in both **Python** and **MATLAB** here:  
👉 [Semi-Supervised-GMM](https://github.com/Shu1L0n9/Semi-Supervised-GMM)

We hope this helps the community grow and explore more in this area! 🚀
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
