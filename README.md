# Carla-Lane-Detection-Dataset-Generation
As part of a project in our university, it was our task to implement an agent in CARLA-Simulator, which automomously collects image and label data to generate a dataset.
This can be used later to train a Deep Convolutional Neural Network, which is able to detect lanemarkings on a road.

The overall project is split up into two parts:

- The first part covers how to create and generate a dataset. It's what this repository is used for. 
- The second part covers the training and testing of a Deep Convolutional Neural Network, which is able to detect lanemarkings on a road. It's what [this repository](https://github.com/Glutamat42/Ultra-Fast-Lane-Detection) is used for.

## Project overview
This repository consists of 2 parts:
- Collect data in CARLA by executing `fast_lane_detection.py`
- Generate a dataset with the collected data `dataset_generator.py`

## Further documentation
For Install / getting started / HOWTOs / etc please see the **full documentation** in the docs directory.
