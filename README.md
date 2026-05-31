<div align="center">

# 🔮 PuriLight

**A Lightweight Shuffle and Purification Framework for Monocular Depth Estimation**

🎉 **Accepted by European Conference on Artificial Intelligence (ECAI 2025)**

[![arXiv](https://img.shields.io/badge/arXiv-2602.11066-b31b1b?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2602.11066)
[![DOI](https://img.shields.io/badge/DOI-10.3233/FAIA251195-blue?logo=doi&logoColor=white)](https://ebooks.iospress.nl/doi/10.3233/FAIA251195)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-green?logo=python&logoColor=white)](https://www.python.org/)

✨ Yujie Chen, Li Zhang, Xiaomeng Chu and Tian Zhang ✨

</div>

---

<div align="center">

# 🚀 PuriLightX

### Towards More Lightweight and Edge-Aware Framework for Self-Supervised Monocular Depth Estimation

📄 *This repository hosts the open-source code for PuriLightX, an upgraded version of our previous work PuriLight. The accompanying paper is currently under review.*

</div>

---

## 📑 Table of Contents

- [Installation](#-installation)
- [Datasets](#-datasets)
- [Pretrained Weights](#-pretrained-weights)
- [Quick Start](#-quick-start)
- [Evaluation](#-evaluation)
- [Single Image Inference](#-single-image-inference)
- [Training](#-training)
- [Results](#-results)
- [Applications](#-applications)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

---

## 🛠️ Installation

### Prerequisites

- Python == 3.10
- PyTorch == 2.3.1
- CUDA 11.8 (recommended)

### Setup env

```bash
# Create conda environment
conda create -n purilightx python=3.10 -y
conda activate purilightx

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

---

## 📦 Datasets

### KITTI

Please follow the instructions in [Monodepth2](https://github.com/nianticlabs/monodepth2) to download and prepare the KITTI dataset.

### Cityscapes

Please follow the instructions in [Manydepth](https://github.com/nianticlabs/manydepth) to download and prepare the Cityscapes dataset.

---

## 🏋️ Pretrained Weights

We provide pretrained weights for evaluation:

| Model         | Dataset    | Resolution | Weights                      |
| ------------- | ---------- | ---------- | ---------------------------- |
| PuriLightX-lp | KITTI      | 640×192    | `weights/kitti_lp_640x192/`  |
| PuriLightX-lf | KITTI      | 640×192    | `weights/kitti_lf_640x192/`  |
| PuriLightX-lp | KITTI      | 1024×320   | `weights/kitti_lp_1024x320/` |
| PuriLightX-lf | KITTI      | 1024×320   | `weights/kitti_lf_1024x320/` |
| PuriLightX-lp | Cityscapes | 512×192    | `weights/cs/`                |
| PuriLightX-lf | Cityscapes | 512×192    | `weights/cs_lf/`             |

> 🚧 Pretrained backbone weights (for training from scratch) will be released soon.

---

## ⚡ Quick Start

The quickest way to get started with PuriLightX:

```bash
# 1. Setup environment
conda create -n purilightx python=3.10 -y && conda activate purilightx
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt



# 2. Run inference on a single image
python test_simple.py \
    --image_path /path/to/image.jpg \
    --load_weights_folder weights/kitti_lp_640x192 \
    --model purilightx-lp
```

## 📊 Evaluation

### KITTI Eigen Split (640×192)

```bash
# PuriLightX-lp
python evaluate_depth.py \
    --load_weights_folder weights/kitti_lp_640x192 \
    --data_path /path/to/kitti \
    --model purilightx-lp \
    --eval_split eigen \
    --png

# PuriLightX-lf
python evaluate_depth.py \
    --load_weights_folder weights/kitti_lf_640x192 \
    --data_path /path/to/kitti \
    --model purilightx-lf \
    --eval_split eigen \
    --png
```

### KITTI Eigen Split (1024×320)

```bash
# PuriLightX-lp
python evaluate_depth.py \
    --load_weights_folder weights/kitti_lp_1024x320 \
    --data_path /path/to/kitti \
    --model purilightx-lp \
    --eval_split eigen \
    --height 320 \
    --width 1024 \
    --png

# PuriLightX-lf
python evaluate_depth.py \
    --load_weights_folder weights/kitti_lf_1024x320 \
    --data_path /path/to/kitti \
    --model purilightx-lf \
    --eval_split eigen \
    --height 320 \
    --width 1024 \
    --png
```

### Cityscapes

```bash
# PuriLightX-lp
python evaluate_depth.py \
    --load_weights_folder weights/cs \
    --data_path /path/to/cityscapes \
    --model purilightx-lp \
    --eval_split cityscapes \
    --height 192 \
    --width 512

# PuriLightX-lf
python evaluate_depth.py \
    --load_weights_folder weights/cs_lf \
    --data_path /path/to/cityscapes \
    --model purilightx-lf \
    --eval_split cityscapes \
    --height 192 \
    --width 512
```

---

## 🖼️ Single Image Inference

```bash
python test_simple.py \
    --image_path /path/to/image.jpg \
    --load_weights_folder weights/kitti_lp_640x192 \
    --model purilightx-lp
```

This saves:

- `*_disp.npy` - predicted disparity as numpy array
- `*_disp.jpeg` - colormapped depth visualization

---

## 🔥 Training

> 🚧 **Training code is coming soon!** We are currently cleaning up and documenting the training pipeline for release. Stay tuned!

In the meantime, you can refer to the training scripts from [Monodepth2](https://github.com/nianticlabs/monodepth2) as a reference, as our training pipeline is built upon it.

---

## 📈 Results

### KITTI Eigen Split

| Model         | Resolution | GFLOPs | Params | abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
| ------------- | ---------- | ------ | ------ | ------- | ------ | ----- | -------- | ----- | ----- | ----- |
| PuriLightX-lp | 640×192    | 7.20G  | 2.39M  | 0.102   | 0.728  | 4.385 | 0.178    | 0.896 | 0.966 | 0.983 |
| PuriLightX-lf | 640×192    | 6.43G  | 2.53M  | 0.102   | 0.730  | 4.466 | 0.180    | 0.894 | 0.965 | 0.983 |
| PuriLightX-lp | 1024×320   | 19.20G | 2.39M  | 0.098   | 0.688  | 4.283 | 0.174    | 0.903 | 0.968 | 0.984 |
| PuriLightX-lf | 1024×320   | 17.15G | 2.53M  | 0.099   | 0.721  | 4.336 | 0.176    | 0.901 | 0.966 | 0.984 |

### Cityscapes

| Model         | Resolution | GFLOPs | Params | abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
| ------------- | ---------- | ------ | ------ | ------- | ------ | ----- | -------- | ----- | ----- | ----- |
| PuriLightX-lp | 512×192    | 5.76G  | 2.39M  | 0.096   | 0.970  | 5.833 | 0.152    | 0.903 | 0.975 | 0.992 |
| PuriLightX-lf | 512×192    | 5.15G  | 2.53M  | 0.098   | 0.984  | 5.791 | 0.153    | 0.901 | 0.976 | 0.992 |

---

## 📝 Citation

If you find our work useful, please consider citing:

```bibtex
@article{chen2026purilight,
  title={PuriLight: A Lightweight Shuffle and Purification Framework for Monocular Depth Estimation},
  author={Chen, Yujie and Zhang, Li and Chu, Xiaomeng and Zhang, Tian},
  journal={arXiv preprint arXiv:2602.11066},
  year={2026}
}
```

---

## 🙏 Acknowledgements

This code is built upon [Monodepth2](https://github.com/nianticlabs/monodepth2) and [Lite-Mono](https://github.com/noahzn/Lite-Mono). We thank the authors for their excellent work.
