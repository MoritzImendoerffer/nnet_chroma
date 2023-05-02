"""
Merging PINN and INVNN

Implementing the INVNN using FrEIA

y[:, 0] for estimation of u
y[:, 1] for estimation of f
seems to work better
"""

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from collections import OrderedDict
from matplotlib import pyplot as plt

#torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


# the deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
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
        out = self.layers(x)
        return out




# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, lb, ub, n_sub=68, n_node=4):
        # number of channels in subnets
        self.n_sub = n_sub

        # number of nodes
        self.n_node = n_node
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        # settings
        self.lambda_1 = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))
        self.lambda_2 = torch.nn.Parameter(torch.tensor([-1.0], requires_grad=True).to(device))

        # construct the INN (not containing any operations so far)
        input_dims = (2, )
        inn = Ff.SequenceINN(*input_dims)

        # append coupling blocks to the sequence of operations
        for k in range(n_node):
            #inn.append(Fm.AllInOneBlock, subnet_constructor=self.subnet_fc)
            # slower but better at version 0.4
            inn.append(Fm.RNVPCouplingBlock, subnet_constructor=self.subnet_fc)


        self.dnn = inn.to(device)

        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        # gamma = decaying factor
        self.scheduler_bfgs = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        # self.optimizer_1 = torch.optim.LBFGS(
        #     self.dnn.parameters(),
        #     lr=1,
        #     max_iter=50000,
        #     max_eval=50000,
        #     history_size=50,
        #     tolerance_grad=1e-5,
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        # )

        # TODO: not sure if multiple optimizers are a benefit
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        #self.optimizer_Adam_1 = torch.optim.Adam(self.dnn.parameters(), lr=0.001)

        self.scheduler_adam = ReduceLROnPlateau(self.optimizer_adam, mode='min', factor=0.1, patience=10, verbose=True)
        #self.optimizer_Adam = torch.optim.SGD(self.dnn.parameters(), lr=0.05, momentum=0.9)
        #self.optimizer_Adam_1 = torch.optim.SGD(self.dnn.parameters(), lr=0.05, momentum=0.9)
        self.iter = 0

    def subnet_fc(self, dims_in, dims_out):
        """Return a feed-forward subnetwork, to be used in the coupling blocks below"""

        n = self.n_sub
        subnet = torch.nn.Sequential(torch.nn.Linear(dims_in, n), torch.nn.Tanh(),
                                     torch.nn.Linear(n, n), torch.nn.Tanh(),
                                     torch.nn.Linear(n, dims_out))
        return subnet

    def net_u(self, x, t):
        u, log_jac_det = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        uhats = self.net_u(x, t).T
        # This is a source for error because the indices are hardcoded in two different functions
        # y values
        # TODO find better way to define the indices
        # I accidentially made the approach better
        u = uhats[:, 1]

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

    def mse_loss_u(self):
        # training so that the output 2 is equal the time
        uhats = self.net_u(self.x, self.t)

        # y values
        u_pred = uhats[:, 0]
        u_pred = u_pred[:, None]

        loss = torch.mean((self.u - u_pred) ** 2)
        return loss

    def mse_loss_f(self):
        # training so that the output 2 is equal the time
        f_pred = self.net_f(self.x, self.t)
        loss = torch.mean(f_pred ** 2)
        return loss

    def loss_func(self):
        """For local optimization using BFGS"""

        loss_u = self.mse_loss_u()
        loss_f = self.mse_loss_f()
        self.optimizer.zero_grad()
        loss_u.backward(retain_graph=True)
        loss_f.backward(retain_graph=True)
        self.scheduler_bfgs.step(loss_u + loss_f)

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'BFGS: It: %d, Loss u: %.3e, Loss f: %.3e, Lambda_1: %.6f, Lambda_2: %.6f' %
                (
                    self.iter,
                    loss_u.item(),
                    loss_f.item(),
                    self.lambda_1.item(),
                    torch.exp(self.lambda_2).item()
                )
            )
        return loss_u + loss_f

    def train(self, nIter):
        self.dnn.train()
        for epoch in range(nIter):
            loss_u = self.mse_loss_u()
            loss_f = self.mse_loss_f()

            # Backward and optimize
            self.optimizer_adam.zero_grad()
            #self.optimizer_Adam_1.zero_grad()
            loss_u.backward(retain_graph=True)

            loss_f.backward(retain_graph=True)
            self.optimizer_adam.step()
            self.scheduler_adam.step(loss_u + loss_f)
            #self.optimizer_Adam_1.step()

            if epoch % 100 == 0:
                print(
                    'Epoch: It: %d, Loss u: %.3e, Loss f: %.3e, Lambda_1: %.6f, Lambda_2: %.6f' %
                    (
                        epoch,
                        loss_u.item(),
                        loss_f.item(),
                        self.lambda_1.item(),
                        torch.exp(self.lambda_2).item()
                    )
                )
        print('Epoch ended')

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

