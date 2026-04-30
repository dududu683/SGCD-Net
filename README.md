# **Task-Driven Underwater Image Enhancement for Visual Measurement: A Statistical Prior Guided Diffusion Model**
## About
This repository provides the complete training and inference code for the proposed statistical prior guided diffusion model. The method enhances degraded underwater images by embedding color statistical priors (mean, variance, skewness, kurtosis) into a conditional diffusion probabilistic framework, specifically designed to support downstream visual measurement tasks (edge detection, keypoint matching).

## Requirements
Before executing the code, make sure all required dependencies listed herein are installed.
```
torch==2.2.1+cu121  
torchvision==0.17.1+cu121  
torch_geometric==2.5.0  
torch_scatter==2.1.2+pt22cu121  
torch_sparse==0.6.18+pt22cu121  
numpy==1.26.4  
Pillow==10.3.0  
tqdm==4.66.4  
warmup_scheduler==0.3  
scikit-image==0.23.2  
torchinfo==1.8.0  
opencv-python==4.9.0.80
```
You can install all dependencies at once via `pip install -r requirements.txt`.

##  Datasets
All datasets utilized in this study are publicly available, and the links to access them are provided below:
UIEB : https://li-chongyi.github.io/proj_benchmark.html
UFO-120 :  http://irvlab.cs.umn.edu/resources/ufo-120-dataset

## Data Preparation
The training requires a paired dataset of underwater degraded images and their corresponding high-quality references (e.g., UIEB). Organize your dataset as:
```
data/
├── train/
│   ├── degraded/   # e.g., UIEB raw images
│   └── target/     # e.g., UIEB reference images
├── val/            # optional
│   ├── degraded/
│   └── target/
└── test/           # for inference
    └── images/
```

## Train
- Before training the model, you need to download the dataset and place it in  `dataset/train`.
- Start training with the following command: `python train.py`
- Then, locate the trained model in  `checkpoint/` .
##  Inference
- To enhance underwater images using a trained model, set the paths in `inference.py` and run:.
```
python inference.py
```
## Acknowledgements

This work is supported by the National College Student Statistical Modeling Competition. The codebase builds upon open-source diffusion model implementations and underwater image enhancement benchmarks (UIEB, UFO-120).
**Contact**[2071965940@qq.com]
