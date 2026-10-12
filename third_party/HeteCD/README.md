## Introduction
This is a repository about the paper "HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection"[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0924271625001066).

## :speech_balloon: Requirements

```
Python 3.8.0
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.2

Please see `requirements.txt` for all the other requirements.

```

## :speech_balloon: Train

1. Download the pre-trained model from Baidu Cloud:  
   [Download Link](https://pan.baidu.com/s/1g9v0rTp1nn4LC5dGTtvDBw) (password: `open`)

2. Modify the path to the pre-trained weights in the configuration.

3. Run:
   python train.py

## :speech_balloon: Evaluate
1. Download the trained model from Baidu Cloud:  
   [Download Link](https://pan.baidu.com/s/1gP0c7uu5W3QDVzSOdarCiw) (password: `open`)


2. Run:
   python predict.py

## :speech_balloon: Dataset Preparation

The XiangAn dataset can be downloaded here:
[Download Link](https://pan.baidu.com/s/1S3nu7Hf8DV1Leu_JVkarig) (password: `open`)

## :speech_balloon: License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## :speech_balloon: Citation

If you use this code for your research, please cite our paper:

```
@article{JING2025317,
title = {HeteCD: Feature Consistency Alignment and difference mining for heterogeneous remote sensing image change detection},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {223},
pages = {317-327},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.03.008},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625001066},
author = {Wei Jing and Haichen Bai and Binbin Song and Weiping Ni and Junzheng Wu and Qi Wang},
keywords = {Heterogeneous change detection, Optical remote sensing image, Synthetic aperture radar image, Heterogeneous feature spaces align, 3D spatio-temporal attention difference},
abstract = {Optical change detection is limited by imaging conditions, hindering real-time applications. Synthetic Aperture Radar (SAR) overcomes these limitations by penetrating clouds and being unaffected by lighting, enabling all-weather monitoring when combined with optical data. However, existing heterogeneous change detection datasets lack complexity, focusing on single-scene targets. To address this gap, we introduce the XiongAn dataset, a novel urban architectural change dataset designed to advance heterogeneous change detection research. Furthermore, we propose HeteCD, a fully supervised heterogeneous change detection framework. HeteCD employs a Siamese Transformer architecture with non-shared weights to effectively model heterogeneous feature spaces and includes a Feature Consistency Alignment (FCA) loss to harmonize distributions and ensure class consistency across bi-temporal images. Additionally, a 3D Spatio-temporal Attention Difference module is incorporated to extract highly discriminative difference information from bi-temporal features. Extensive experiments on the XiongAn dataset demonstrate that HeteCD achieves a superior IoU of 67.50%, outperforming previous state-of-the-art methods by 1.31%. The code will be available at https://github.com/weiAI1996/HeteCD.}
}
```


