# Content-Based Image Retrieval (CBIR) System
CS5330 Project 2

## Introduction
This project implements a Content-Based Image Retrieval (CBIR) system, which allows users to search for similar images based on a target image. The system utilizes various feature extraction techniques and distance metrics to find the most similar images from a database.

## Features
- Baseline Matching: Uses a 7x7 square in the middle of the image as a feature vector with sum-of-squared-difference as the distance metric.
- Histogram Matching: Utilizes normalized color histograms with histogram intersection as the distance metric.
- Multi-histogram Matching: Uses multiple color histograms representing different spatial parts of the image.
- Texture and Color: Combines whole image color histogram and texture histogram, with a custom-designed distance metric.
- Deep Network Embeddings: Utilizes feature vectors obtained from a pre-trained ResNet18 deep network, with customizable distance metrics.
- Custom Design: Allows users to choose specific types of images and design custom feature vectors and distance metrics.

## Extensions
- Inplemented Laws Filter with Color Histogram.
- Trash Can Detector: When given a target trash can image, all the images in the directory are displayed.
- Implemeted HOG Filter.
- Banana Detector: Detects all the bananas in a given image directory.
