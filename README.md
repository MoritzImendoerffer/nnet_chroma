# Invertible Physics Informed Neural Network (IPINN)

This project implements an invertible physics informed neural network using the Burgers equation as a toy example. The implementation is converted from Tensorflow (https://github.com/maziarraissi/PINNs) to PyTorch. The purpose of this project is to provide a starting point for implementing invertible neural networks in physical systems.

The Repo does contain three folders:

pinns_burger ... simple test case for the Burger`s equation
invertible_pinns ... attempt to make pinns_burger invertible for potentially rapit parameter estimation.
pinns_grm ... implementation of the general rate model for protein chromatography as PINN

## Implementation Overview

### General Rate model for Protein Chromatography

This project aims at implementing a Physics Informed Neural Network (PINN) for the General Rate Model (GRM) in protein chromatography. The GRM-PINN is capable of working with different binding models, such as Langmuir and Steric Mass Action models, and takes pore and surface diffusion into account. The implementation also incorporates Dankwerts boundary conditions to ensure the model adheres to the physical constraints of the problem.

The current implementation is at an early stage and requires testing.

#### Introduction

The GRM is based on the following partial differential equations (PDEs):

1. Continuity equation for the mobile phase:

   ∂c_m(x, t) / ∂t = -v ∂c_m(x, t) / ∂x + D_pore ∂²c_m(x, t) / ∂x² - (1 - ε) ∂q(x, t) / ∂t

2. Continuity equation for the solid phase:

   ∂q(x, t) / ∂t = kf(c_m(x, t), q(x, t)) - kr(q(x, t))

where:

- c_m(x, t) is the concentration of the protein in the mobile phase at position x and time t.
- q(x, t) is the concentration of the protein in the solid phase at position x and time t.
- v is the mobile phase velocity.
- ε is the bed porosity.
- D_pore is the pore diffusion coefficient.
- kf and kr are the forward and reverse rate constants for the binding models, respectively.

#### Binding Models

##### Langmuir Binding Model

The Langmuir binding model assumes a homogeneous binding surface with one binding site per protein molecule. The forward and reverse rate constants are given by:

- kf = k_a * c_m(x, t) * (Q_max - q(x, t))
- kr = k_d * q(x, t)

where:

- k_a and k_d are the adsorption and desorption rate constants, respectively.
- Q_max is the maximum capacity of the solid phase.

##### Steric Mass Action (SMA) Binding Model

The SMA binding model considers a heterogeneous binding surface with multiple binding sites per protein molecule. The forward and reverse rate constants are given by:

- kf = k_a * c_m(x, t) * (Q_max - q(x, t)) / (1 + λ * c_m(x, t))
- kr = k_d * q(x, t)

where:

- λ is the steric shielding factor.

#### Boundary Conditions

Dankwerts boundary conditions are used to ensure that the model adheres to the physical constraints of the problem:

1. At the inlet (x = 0):

   - ∂c_m(0, t) / ∂x = 0
   - c_m(0, t) = c_inlet(t)

2. At the outlet (x = L):

   - ∂c_m(L, t) / ∂x = 0

where:

- L is the column length.
- c_inlet(t) is the inlet concentration as a function of time.


#### Usage

1. Prepare your data for simulation. Define the initial conditions, boundary conditions, and model parameters.
2. Instantiate the `GeneralRateModel` class, passing in the necessary parameters and the desired binding model.
3. Use the `simulate()` method on the `GeneralRateModel` instance to obtain the simulated concentration profiles for the mobile and solid phases.

#### Example

Work in progress

#### TODO

Generate test data e.g. using python-cadet
https://github.com/modsim/CADET-Tutorial


### Burger`s Equation
The Burgers equation is used as the initial test case for implementing the physics informed neural network. This implementation is carried out using PyTorch and is divided into two main classes: `DNN` and `PhysicsInformedNN`.

#### DNN (Deep Neural Network)

The `DNN` class implements a simple feedforward deep neural network with a specified number of layers and Tanh activation functions. It receives an input and passes it through the layers to produce the output.

#### PhysicsInformedNN

The `PhysicsInformedNN` class implements the main functionality of the physics informed neural network. It takes input data, boundary conditions, and network architecture as inputs. This class has two main methods:

1. `net_u`: This method computes the output of the neural network for a given input. It uses the DNN class to get the output.
2. `net_f`: This method computes the residual of the physics-informed loss function using the network output and the known physics of the system (Burgers equation in this case).

The training process minimizes the sum of the mean squared errors of the network output (`u_pred`) and the residual of the physics-informed loss function (`f_pred`).

#### Usage

1. Load the data (Burgers equation) from the provided `.mat` file. Source: https://github.com/maziarraissi/PINNs
2. Define the parameters for the problem: number of training samples, network layers, and domain bounds.
3. Create an instance of the `PhysicsInformedNN` class with the training data, network architecture, and domain bounds.
4. Train the model using the `train` method.
5. Use the `predict` method to obtain predictions for new input data.
6. Visualize the results using Matplotlib.

### Current State

Experimenting with implementing in invertible form of the PINNS. It seems, that the solution is not stable and oszillates.

TODO: Test oscillating LR scheduler, and more iterations