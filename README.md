# Photo to Comic Model

This is a machine learning model build with the help of Tensorflow a python machine learning library. It consists of 5 convolution and 5 deconvolution layers of neural network to initially compress the image data and then reconstruct the image into its suitable form.
It takes rgb image of size 512x512 and outputs cartonized image of same size.

## Introduction

This project is aimed to accomplish image to image translation of real world images into its suitable cartoon form. With the help of power of machine learning, it was achieved. It uses autoencoder to compress and reconstruct images. This project was a challenge as autoencoders are usually not used for the purpose of image to image translation. Rather it is mainly used for image denoising. Results of imgetoimge translation was not as good as expected but was really fruitful to learn about the insights of the machine learning.

### Requirements

* python 3.9.15
* tensorflow 2.10.0
* numPy
* matplotlib

## Datasets

Datasets of real face images and its corresponding cartoon images were collected from the kaggle. You can download datasets from this link <https://www.kaggle.com/datasets/defileroff/comic-faces-paired-synthetic-v2>. It contains the pair of face image and its corresponding comic version. Seperate the imges into 'train' image and 'validation' image and also for testing.
