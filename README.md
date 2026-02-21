<div align="center">

<h1><a href="http://arxiv.org/abs/2503.06277">STiL: Semi-supervised Tabular-Image Learning for Comprehensive Task-Relevant Information Exploration in Multimodal Classification (CVPR 2025)</a></h1>

**[Siyi Du](https://scholar.google.com/citations?user=zsOt8MYAAAAJ&hl=en), [Xinzhe Luo](https://scholar.google.com/citations?user=l-oyIaAAAAAJ&hl=en&oi=ao), [Declan P. O'Regan](https://scholar.google.com/citations?user=85u-LbAAAAAJ&hl=en&oi=ao), and [Chen Qin](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=mTWrOqHOqjoC&pagesize=80&sortby=pubdate)** 

![](https://komarev.com/ghpvc/?username=siyi-windSTiL&label=visitors)
![GitHub stars](https://badgen.net/github/stars/siyi-wind/STiL)
[![](https://img.shields.io/badge/license-Apache--2.0-blue)](#License)
[![](https://img.shields.io/badge/arXiv-2503.06277-b31b1b.svg)](https://arxiv.org/abs/2503.06277)

</div>

![TIP](./Images/model.jpg)
<p align="center">Overall framework of STiL. STiL encodes image-tabular data using $\phi$, decomposes modality-shared and -specific information through DCC $\psi$ (a), and outputs predictions via multimodal and unimodal classifiers $f$. STiL generates pseudo-labels for unlabeled data using CGPL (b) and refines them with prototype similarity scores in PGLS (c). (d) Training pathways for labeled and unlabeled data.</p>

This is an official PyTorch implementation for [STiL: Semi-supervised Tabular-Image Learning for Comprehensive Task-Relevant Information Exploration in Multimodal Classification][1]. We built the code based on the code of our prior ECCV 2024 paper [siyi-wind/TIP](https://github.com/siyi-wind/TIP). 

We also include plenty of comparing models in this repository: [SimMatch](http://openaccess.thecvf.com/content/CVPR2022/html/Zheng_SimMatch_Semi-Supervised_Learning_With_Similarity_Matching_CVPR_2022_paper.html), Multimodal SimMatch, [CoMatch](http://openaccess.thecvf.com/content/ICCV2021/html/Li_CoMatch_Semi-Supervised_Learning_With_Contrastive_Graph_Regularization_ICCV_2021_paper.html), Multimodal CoMatch, [FreeMatch](https://arxiv.org/abs/2205.07246), Multimodal FreeMatch, [MMatch](https://ieeexplore.ieee.org/abstract/document/9733884), and [Co-training](https://dl.acm.org/doi/abs/10.1145/279943.279962) (Please go to the paper to find the detailed information of these models).

Concact: s.du23@imperial.ac.uk (Siyi Du)

Share us a :star: if this repository does help. 

## Updates
[**12/03/2025**] The arXiv paper and the code are released. 

[**21/02/2026**] We have a new paper accepted at ICLR 2026, which proposes an inference-time dynamic modality selection framework (DyMo) for various missing data scenarios across multiple modalities. Please check [this repository](https://github.com/siyi-wind/DyMo) for details. 

## Our Multimodal Learning Research Line
This repository is part of our research line on multimodal learning.

- **TIP (ECCV2024)**: An image-tabular pre-training framework for intra-modality missingness ([siyi-wind/TIP](https://github.com/siyi-wind/TIP))

- **STiL (CVPR 2025, this work)**: A semi-supervised image-tabular framework for modality heterogeneity and limited labeled data ([siyi-wind/STiL](https://github.com/siyi-wind/STiL))

- **DyMo (ICLR 2026)**: An inference-time dynamic modality selection framework for missing modality ([siyi-wind/DyMo](https://github.com/siyi-wind/DyMo))



## Contents
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training & Testing](#training--testing)
- [Checkpoints](#checkpoints)
- [Lisence & Citation](#lisence--citation)
- [Acknowledgements](#acknowledgements)

## Requirements
This code is implemented using Python 3.9.15, PyTorch 1.11.0, PyTorch-lighting 1.6.4, CUDA 11.3.1, and CuDNN 8.

```sh
cd STiL/
conda env create --file environment.yaml
conda activate stil
```

## Data Preparation
Download DVM data from [here][2]

Apply for the UKBB data [here][3]

### Preparation
We conduct the same data preprocessing process as [siyi-wind/TIP](https://github.com/siyi-wind/TIP).

## Training & Testing

### Training
```sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01 exp_name=train evaluate=True checkpoint={YOUR_PRETRAINED_CKPT_PATH}
```

### Testing
```sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01 exp_name=test test=True checkpoint={YOUR_TRAINED_CKPT_PATH}
```

## Checkpoints

Task | 1% labeled | 10% labeled
--- | :---: | :---: 
Car model prediction (DVM) | [Download](https://drive.google.com/drive/folders/1nWbmPOLdnoTDnr4zS56HPFkt-k656jM4?usp=sharing)  | [Download](https://drive.google.com/drive/folders/1F-hV_uh5BPjMc-cKY0hixsI7irWfRk3S?usp=sharing)
CAD classification (Cardiac) | [Download](https://drive.google.com/drive/folders/10ddCKsMpsSQEyt6F-c1qtM9PKXTM9xiR?usp=sharing) | [Download](https://drive.google.com/drive/folders/1Es9CkwxGz7jnU4RtSFcl9fhN5h0Y6y5q?usp=sharing)
Infarction classification (Cardiac) | [Download](https://drive.google.com/drive/folders/139MEzfdXvHg7lSRZKjbiO5mkd1mw8SwF?usp=sharing) | [Download](https://drive.google.com/drive/folders/11SXYjZQUWa6d5btK0Kh18d94L4Sr0369?usp=sharing)


## Lisence & Citation
This repository is licensed under the Apache License, Version 2.

If you use this code in your research, please consider citing:

```text
@inproceedings{du2025stil,
  title={{STiL}: Semi-supervised Tabular-Image Learning for Comprehensive Task-Relevant Information Exploration in Multimodal Classification},
  author={Du, Siyi and Luo, Xinzhe and O'Regan, Declan P. and Qin, Chen},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR) 2025},
  year={2025}}
```

## Acknowledgements
We would like to thank the following repositories for their great works:
* [PIBD](https://github.com/zylbuaa/PIBD)
* [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging)
* [BLIP](https://github.com/salesforce/BLIP)



[1]: http://arxiv.org/abs/2503.06277
[2]: https://deepvisualmarketing.github.io/
[3]: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access