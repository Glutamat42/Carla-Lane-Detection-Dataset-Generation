# UFLD Documentation

Our project consists of two parts. One part is about generating datasets from CARLA, which is topic of this documentation.
The other part is about a neural network, that is able to detect lanemarkings on a road. The documentation about this part can be found [here](https://mi-project.markus-heck.dev/ld_docs/index.html).

## Project overview

The project is split up into two parts:

- The first part covers how to collect labeled images from CARLA.
- The second part covers how to generate a dataset from the collected labeled images. 

## Datasets

In order to train a Deep Convolutional Neural Network, we have to create a dataset, containing images and labels, which indicate the lanemarkings on a road. For this purpose, the dataset has to meet some requirements to be read by the Deep Convolutional Neural Network. If the dataset doesn't fulfill the specification exactly it will fail.

## Collecting data from CARLA

CARLA is one way to get training and test data. Sadly CARLA doesn't provide the required information over its API. As a
result it's quite complicated to generate a dataset from CARLA. We wrote some scripts which make this possible. The
scripts are located in the [scripts](https://github.com/Glutamat42/Carla-Lane-Detection-Dataset-Generation) folder. If you want to use them correctly,
look [on this HOWTO](howto/generate_dataset_from_carla.md) which explains their usage in detail.

## Build documentation

The project documentation was set up using sphinx. In order to generate and build the html file, make sure you have sphinx installed. After that, execute the following steps:

1. Change to `docs/`
2. Run `make html` in the terminal


```{toctree}
---
maxdepth: 1
caption: Contents
---
./howto/howtos
```


   