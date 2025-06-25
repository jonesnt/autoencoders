# Purpose
This repository is meant to be the building blocks for the training of a neural network which will classify Scanning Tunneling Microscopy images.

# Methodolgy
The code in the *modular_autoencoder* directory is meant to simplify the process of training the neural network using multiple methods of artificial image generation and multiple training models. The *training_manager.py* file is the entry point for training our neural network and when invoked via the command line, the user is able to specify the learning rate of the neural net, the number of epochs for training, and the number of images which should be created and used as the training set.