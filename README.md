# Purpose
This repository is meant to be the building blocks for the training of a neural network which will classify Scanning Tunneling Microscopy images.

# Methodolgy
The code in the *modular_autoencoder* directory is meant to simplify the process of training the neural network using multiple methods of artificial image generation and multiple training models. The *training_manager.py* file is the entry point for training our neural network and when invoked via the command line, the user is able to specify the learning rate of the neural net, the number of epochs for training, and the number of images which should be created and used as the training set.

# Initialization Instructions
1. Once logged in to the Theia environment, ensure that you have set up the git environment using the following command:
```
    git clone https://github.com/jonesnt/autoencoders.git
```
**YOU ONLY NEED TO DO THIS ONCE**

2. Install Anaconda with the instructions [here](https://youtu.be/sU2mXjOB-fA?si=WFYc05ljOxJ-y_mM) and ensure that the install script configures your path.

**YOU ONLY NEED TO DO THIS ONCE**

3. Navigate to the appropriarte directory by typing
```
cd /autencoders/modular_autoencoder
```

4. Modify the *autoencoder.sh* file to contain the instructions you want for the training process. Note that you can modify the script to send you an email notification whenever the training process begins, fails, or finishes.

5. Run the trainer by typing
```
sbatch ./autoencoder.sh
```
