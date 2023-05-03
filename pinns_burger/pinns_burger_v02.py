"""
Invertible Physics Informed Neural Network (IPINN)

This project implements an invertible physics informed neural network using
the Burgers equation as a toy example. The purpose of this project is to
provide a starting point for implementing invertible neural networks in physical systems.
"""

import torch
torch.manual_seed(0)
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau



device = "cuda" if torch.cuda.is_available() else "cpu"


class DNN(torch.nn.Module):
    """
    Deep Neural Network (DNN) class.

    This class implements a simple feedforward deep neural network with a specified
    number of layers and Tanh activation functions.
    """

    def __init__(self, layers):
        """
        Initialize the DNN class.

        Args:
            layers (list): List of integers defining the number of neurons in each layer.
        """
        super(DNN, self).__init__()

        layer_list = list()
        layer_dict = OrderedDict()
        for i in range(len(layers) - 2):
            lname = f"layer_{i:d}"
            layer_dict[lname] = torch.nn.Linear(layers[i], layers[i + 1])

            aname = f"activation_{i:d}"
            layer_dict[aname] = torch.nn.Tanh()

        lname = f"layer_{len(layers) - 2: d}"
        layer_dict[lname] = torch.nn.Linear(layers[-2], layers[-1])

        # deploy layers
        self.layers = torch.nn.Sequential(layer_dict)


    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        out = self.layers(x)
        return out


class PhysicsInformedNN():
    """
    Physics Informed Neural Network (PINN) class.

    This class implements the main functionality of the physics informed neural network.
    It takes input data, boundary conditions, and network architecture as inputs.
    """

    def __init__(self, X, u, layers, lb, ub):
        """
        Initialize the PhysicsInformedNN class.

        Args:
            X (numpy.array): Input features (x, t) for training.
            u (numpy.array): Output values for training.
            layers (list): List of integers defining the number of neurons in each layer.
            lb (numpy.array): Lower bounds of the input features.
            ub (numpy.array): Upper bounds of the input features.
        """
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        # settings
        self.lambda_1 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True).to(device))
        self.lambda_2 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True).to(device))

        # deep neural networks
        self.dnn = DNN(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

        # add the learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer_Adam, mode='min', factor=0.1, patience=10, verbose=True)

    def net_u(self, x, t):
        """
        Compute the output of the neural network for a given input.

        Args:
            x (torch.Tensor): Spatial input tensor.
            t (torch.Tensor): Time input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """
        Compute the residual of the physics-informed loss function.

        Args:
            x (torch.Tensor): Spatial input tensor.
            t (torch.Tensor): Time input tensor.

        Returns:
            torch.Tensor: Residual of the physics-informed loss function.
        """
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def loss_func(self):
        """
        Loss function for local optimization using BFGS.

        Returns:
            torch.Tensor: Loss value for the current model state.
        """
        u_pred = self.net_u(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)
        self.optimizer.zero_grad()
        loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'A: Loss: %e, l1: %.5f, l2: %.5f' %
                (
                    loss.item(),
                    self.lambda_1.item(),
                    torch.exp(self.lambda_2.detach()).item()
                )
            )
        return loss

    def train(self, nIter):
        """
        Train the physics informed neural network.

        Args:
            nIter (int): Number of training iterations.
        """
        self.dnn.train()
        for epoch in range(nIter):
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()
            before_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            after_lr = self.optimizer.param_groups[0]["lr"]
            if epoch % 100 == 0:
                print(
                    'B: It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, LR: %.6f' %
                    (
                        epoch,
                        loss.item(),
                        self.lambda_1.item(),
                        torch.exp(self.lambda_2).item(),
                        after_lr
                    )
                )

        print('Epoch ended')
        # Backward and optimize
        self.optimizer.step(self.loss_func)
        print('Loss func called')

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f

