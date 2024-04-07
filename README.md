# Code implementation for Enhancing Aerial Semantic Segmentation with Feature Aggregation Network for DeepLabV3+

## Table of Contents

- [Abstract](#abstract)
- [Model Architecture](#modelarchitecture)
- [Training](#training)
- [Testing](#testing)
- [Results](#result)

## Abstract

As a cutting-edge deep encoder-decoder architecture, DeepLabV3+ has been realized as a state-of-the-art solution for image segmentation. Furthermore, DeepLabV3+ has great
potential for semantic segmentation of aerial images captured by
unmanned aerial vehicles (UAVs) for aerial and remote sensing
applications. This is thanks to an Atrous Spatial Pyramid Pooling
(ASPP) block deployed in its encoder with multiple atrous
convolutional layers to enrich diversified feature extraction and
learning efficiency. However, the encoder-decoder architecture of
DeepLabV3+ has some limitations, including the lack of infor-
mation during the upsampling process and some inappropriate
customizations that cause incorrect segmentation. To address
these shortcomings, we introduce an efficient architecture with
a novel Feature Aggregation Network (FAN), which facilitates
the extraction of features across multiple scales and stages.
Concurrently, we apply some adaptive upgrades to the ASPP
block, involving a new set of dilation factors that are adept at
accommodating low-resolution inputs. As a result, our enhanced
remote sensing segmentation model achieves significant perfor-
mance gains when being evaluated on a real-world dataset: global
accuracy improves by at least 5.39%, mean intersection-over-
union (IoU) increases by 10.97%, and mean boundary-F1-score
(BFScore) improves by 11.3%. These advancements lead to more
precise identification of urban classes, resulting in heightened
accuracy in the segmentation task.

## ModelArchitecture

![image](https://github.com/giangcutebhc2/GSRL/assets/104675768/e38ab548-673f-4a0e-9c79-26d05c9865fb)

## Training

To train the proposed model, please run the main file:
```sh
main.m
```
The architecture of the model can be obtained through the command:
```sh
load('model_architecture.mat')
```
### Testing

To test our trained model, please run the test file:
```sh
test.m
```
The model with the trained weights can be obtained through the commands:
```sh
load('trained_model.mat');
net = trainednetInfo{1,1};
```

## Results
![image](https://github.com/giangcutebhc2/GSRL/assets/104675768/1a2a3218-532d-4635-a1c4-a82785f679b5)
![image](https://github.com/giangcutebhc2/GSRL/assets/104675768/485b382e-600a-4f55-b654-92e370c022c6)
