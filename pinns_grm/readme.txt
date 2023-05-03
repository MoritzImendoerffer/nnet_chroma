# General Rate Model (GRM) for Protein Chromatography

This project provides an implementation of the General Rate Model (GRM) for protein chromatography, which is widely used for simulating and optimizing chromatographic separations of proteins. The GRM incorporates different binding models, such as Langmuir and Steric Mass Action (SMA), and considers pore and surface diffusion as well as convective transport.

## Introduction

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

## Binding Models

### Langmuir Binding Model

The Langmuir binding model assumes a homogeneous binding surface with one binding site per protein molecule. The forward and reverse rate constants are given by:

- kf = k_a * c_m(x, t) * (Q_max - q(x, t))
- kr = k_d * q(x, t)

where:

- k_a and k_d are the adsorption and desorption rate constants, respectively.
- Q_max is the maximum capacity of the solid phase.

### Steric Mass Action (SMA) Binding Model

The SMA binding model considers a heterogeneous binding surface with multiple binding sites per protein molecule. The forward and reverse rate constants are given by:

- kf = k_a * c_m(x, t) * (Q_max - q(x, t)) / (1 + λ * c_m(x, t))
- kr = k_d * q(x, t)

where:

- λ is the steric shielding factor.

## Boundary Conditions

Dankwerts boundary conditions are used to ensure that the model adheres to the physical constraints of the problem:

1. At the inlet (x = 0):

   - ∂c_m(0, t) / ∂x = 0
   - c_m(0, t) = c_inlet(t)

2. At the outlet (x = L):

   - ∂c_m(L, t) / ∂x = 0

where:

- L is the column length.
- c_inlet(t) is the inlet concentration as a function of time.

## Requirements

- Python 3.7 or later
- NumPy
- SciPy
- PyTorch

## Usage

1. Prepare your data for simulation. Define the initial conditions, boundary conditions, and model parameters.
2. Instantiate the `GeneralRateModel` class, passing in the necessary parameters and the desired binding model.
3. Use the `simulate()` method on the `GeneralRateModel` instance to obtain the simulated concentration profiles for the mobile and solid phases.

## Example

Please refer to the `grm.py

## TODO

Generate test data e.g. using python-cadet
https://github.com/modsim/CADET-Tutorial