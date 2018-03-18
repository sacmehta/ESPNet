#  ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

This repository contains the source code of our paper, ESPNet.


## Structure of this repository
This repository is organized as:
* [train](/train/) This directory contains the source code for trainig the ESPNet-C and ESPNet models.
* [test](/test/) This directory contains the source code for evaluating our model on RGB Images.
* [pretrained] (/pretrained/) This directory contains the pre-trained models on the CityScape dataset
  * [encoder](/pretrained/encoder/) This directory contains the pretrained **ESPNet-C** models
  * [decoder](/pretrained/decoder/) This directory contains the pretrained **ESPNet** models


## Performance on the CityScape dataset

Our model ESPNet achives an class-wise mIOU of **60.336** and category-wise mIOU of **82.178** on the CityScapes validation dataset and runs at 
* 112 fps on the NVIDIA TitanX (30 fps faster than [ENet](https://arxiv.org/abs/1606.02147))
* 9 FPS on TX2
* With the same number of parameters as [ENet](https://arxiv.org/abs/1606.02147), our model is **2%** more accurate

## Pre-requisite

To run this code, you need to have following libraries:
* [OpenCV](https://opencv.org/) - We tested our code with version > 3.0.
* [PyTorch](http://pytorch.org/)

We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html). We have tested our code on Ubuntu 16.04.
