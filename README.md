#  ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation

This repository contains the source code of our paper, [ESPNet](https://arxiv.org/abs/1803.06815).


## Structure of this repository
This repository is organized as:
* [train](/train/) This directory contains the source code for trainig the ESPNet-C and ESPNet models.
* [test](/test/) This directory contains the source code for evaluating our model on RGB Images.
* [pretrained](/pretrained/) This directory contains the pre-trained models on the CityScape dataset
  * [encoder](/pretrained/encoder/) This directory contains the pretrained **ESPNet-C** models
  * [decoder](/pretrained/decoder/) This directory contains the pretrained **ESPNet** models


## Performance on the CityScape dataset

Our model ESPNet achives an class-wise mIOU of **60.336** and category-wise mIOU of **82.178** on the CityScapes test dataset and runs at 
* 112 fps on the NVIDIA TitanX (30 fps faster than [ENet](https://arxiv.org/abs/1606.02147))
* 9 FPS on TX2
* With the same number of parameters as [ENet](https://arxiv.org/abs/1606.02147), our model is **2%** more accurate

## Performance on the CamVid dataset

Our model achieves an mIOU of 55.64 on the CamVid dataset.

| Model | mIOU | Class avg. | 
| -- | -- | -- |
| ENet | 51.3 | 68.3 | 
| SegNet | 55.6 | 65.2 | 
| ESPNet | 55.64 | 68.30 | 
| -- | -- | -- |

## Pre-requisite

To run this code, you need to have following libraries:
* [OpenCV](https://opencv.org/) - We tested our code with version > 3.0.
* [PyTorch](http://pytorch.org/)

We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html). We have tested our code on Ubuntu 16.04.

## Citation
If ESPNet is useful for your research, then please cite our paper.
```
@article{mehta2018espnet,
  title={ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation},
  author={Sachin Mehta, Mohammad Rastegari, Anat Caspi, Linda Shapiro, and Hannaneh Hajishirzi},
  journal={arXiv preprint arXiv:1803.06815},
  year={2018}
}
```
