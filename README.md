# Perceptual Assessment and Optimization of HDR Image Rendering

## Introduction
This repository contains the official implementation of the paper ["Perceptual Assessment and Optimization of HDR Image Rendering
"](https://arxiv.org/abs/2310.12877) by Peibei Cao, Rafal K. Mantiuk, and Kede Ma, IEEE Conference on Computer Vision and Pattern Recognition, 2024.
## Prerequisites
* python>=3.6

## As assessment metric
### Usage:



## As loss function
### Usage:
```ruby
from HDRloss import hdrLoss
D = hdrLoss()
HDR_Loss = D(X, Y)
HDR_Loss.backward()
```

