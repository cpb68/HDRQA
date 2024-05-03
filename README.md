# Perceptual Assessment and Optimization of HDR Image Rendering

## Introduction
This repository contains the official implementation of the paper ["Perceptual Assessment and Optimization of HDR Image Rendering
"](https://arxiv.org/abs/2310.12877) by Peibei Cao, Rafal K. Mantiuk, and Kede Ma, IEEE Conference on Computer Vision and Pattern Recognition, 2024.
## Prerequisites
* python>=3.6

## As assessment metric
### Usage:
* Please put reference HDR images in  ``./assessment/image/ref/``, and test HDR images in ``./assessment/image/test/``.
* Please put the name of test HDR images in ``./assessment/samples.txt``.
* To evaluate the test HDR image: 
```bash
python ./assessment/HDRMetric.py
```
**Please make sure the LDR sequence is perfectly aligned with the reference sequence.**

## As loss function
### Usage:
```ruby
from HDRloss import hdrLoss
D = hdrLoss()
HDR_Loss = D(X, Y)
HDR_Loss.backward()
```
**Please contact peibeicao2-c@my.cityu.edu.hk if you have any problem with the code.**

## Citation
```
@article{Cao2023Perceptual,
  title={Perceptual Assessment and Optimization of High Dynamic Range Image Rendering},
  author={Peibei Cao and Rafał K. Mantiuk and Kede Ma},
  journal={ArXiv},
  year={2023},
  volume={abs/2310.12877},
  url={https://api.semanticscholar.org/CorpusID:264306134}
}
```
