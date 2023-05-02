# Invertible Physics Informed Neural Network (IPINN)

This project implements an invertible physics informed neural network using the Burgers equation as a toy example. The implementation is converted from Tensorflow (https://github.com/maziarraissi/PINNs) to PyTorch. The purpose of this project is to provide a starting point for implementing invertible neural networks in physical systems.



## Implementation Overview

The Burgers equation is used as the initial test case for implementing the physics informed neural network. This implementation is carried out using PyTorch and is divided into two main classes: `DNN` and `PhysicsInformedNN`.

### DNN (Deep Neural Network)

The `DNN` class implements a simple feedforward deep neural network with a specified number of layers and Tanh activation functions. It receives an input and passes it through the layers to produce the output.

### PhysicsInformedNN

The `PhysicsInformedNN` class implements the main functionality of the physics informed neural network. It takes input data, boundary conditions, and network architecture as inputs. This class has two main methods:

1. `net_u`: This method computes the output of the neural network for a given input. It uses the DNN class to get the output.
2. `net_f`: This method computes the residual of the physics-informed loss function using the network output and the known physics of the system (Burgers equation in this case).

The training process minimizes the sum of the mean squared errors of the network output (`u_pred`) and the residual of the physics-informed loss function (`f_pred`).

## Usage

1. Load the data (Burgers equation) from the provided `.mat` file. Source: https://github.com/maziarraissi/PINNs
2. Define the parameters for the problem: number of training samples, network layers, and domain bounds.
3. Create an instance of the `PhysicsInformedNN` class with the training data, network architecture, and domain bounds.
4. Train the model using the `train` method.
5. Use the `predict` method to obtain predictions for new input data.
6. Visualize the results using Matplotlib.

## Current State

Experimenting with implementing in invertible form of the PINNS. It seems, that the solution is not stable and oszillates.