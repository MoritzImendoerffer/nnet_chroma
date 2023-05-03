"""
Implementation of the general rate model for protein chromatography as physics informed neural network
"""

import torch
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = "cuda" if torch.cuda.is_available() else "cpu"

class LangmuirModel:
    def __init__(self, k_ads, k_des):
        self.k_ads = k_ads
        self.k_des = k_des

    def binding(self, c, q):
        return self.k_ads * c - self.k_des * q


class StericMassActionModel:
    def __init__(self, k_a, k_d, nu, sigma, q_max):
        self.k_a = k_a
        self.k_d = k_d
        self.nu = nu
        self.sigma = sigma
        self.q_max = q_max

    def binding(self, c, q):
        lambda_c = (self.q_max - q) / self.sigma
        return self.k_a * (c ** self.nu) * lambda_c - self.k_d * q

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

        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, x):
        out = self.layers(x)
        return out


class GRM_PINN():
    def __init__(self, X, u, layers, lb, ub):
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.dnn = DNN(layers).to(device)

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

        self.scheduler_adam = ReduceLROnPlateau(self.optimizer_Adam, mode='min', factor=0.1, patience=10, verbose=True)
        self.scheduler_bfgs = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    def net_c(self, x, t):
        c = self.dnn(torch.cat([x, t], dim=1))
        return c

    def net_q(self, x, t):
        q = self.dnn(torch.cat([x, t], dim=1))
        return q

    def net_f(self, x, t):
        c = self.net_c(x, t)
        q = self.net_q(x, t)

        c_t = torch.autograd.grad(
            c, t,
            grad_outputs=torch.ones_like(c),
            retain_graph=True,
            create_graph=True
        )[0]
        c_x = torch.autograd.grad(
            c, x,
            grad_outputs=torch.ones_like(c),
            retain_graph=True,
            create_graph=True
        )[0]

        q_t = torch.autograd.grad(
            q, t,
            grad_outputs=torch.ones_like(q),
            retain_graph=True,
            create_graph=True
        )[0]

        k_ads = 1.0 
        k_des = 1.0
        k_p = 1.0
        epsilon = 0.4

        f_c = c_t - k_ads * (1 - epsilon) * (c - q) + k_des * (1 - epsilon) * q
        f_q = q_t - k_p * (c - q)

        return f_c, f_q

    def dankwerts_boundary_conditions(self, x):
        c_x_0 = torch.autograd.grad(
            self.c_inlet, x,
            grad_outputs=torch.ones_like(self.c_inlet),
            retain_graph=True,
            create_graph=True
        )[0]

        bc_0 = self.v_inlet * (self.c_inlet - self.net_u(x, 0)) - self.D_pore * c_x_0
        bc_1 = self.net_u(x, self.ub[1]) - self.c_inlet

        return bc_0, bc_1

    def loss_func(self):
        u_pred = self.net_u(self.x, self.t)
        f_pred_c, f_pred_q = self.net_f(self.x, self.t)

        # Enforce boundary conditions
        bc_0, bc_1 = self.dankwerts_boundary_conditions(self.x)

        loss = (
            torch.mean((self.u - u_pred) ** 2) +
            torch.mean(f_pred_c ** 2) +
            torch.mean(f_pred_q ** 2) +
            torch.mean(bc_0 ** 2) +
            torch.mean(bc_1 ** 2)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.scheduler_bfgs.step(loss)
        after_lr = self.optimizer.param_groups[0]["lr"]
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'A: Loss: %e, l1: %.5f, l2: %.5f, LR: %.6f' %
                (
                    loss.item(),
                    self.lambda_1.item(),
                    torch.exp(self.lambda_2.detach()).item(),
                    after_lr
                )
            )
        return loss

    def train(self, nIter):
        self.dnn.train()
        for epoch in range(nIter):
            self.optimizer_Adam.zero_grad()

            c_pred, q_pred = self.net_c(self.x, self.t), self.net_q(self.x, self.t)
            f_c_pred, f_q_pred = self.net_f(self.x, self.t)

            loss_c = torch.mean((self.u - c_pred) ** 2)
            loss_q = torch.mean((self.u - q_pred) ** 2)
            loss_f_c = torch.mean(f_c_pred ** 2)
            loss_f_q = torch.mean(f_q_pred ** 2)

            loss = loss_c + loss_q + loss_f_c + loss_f_q

            loss.backward()
            self.optimizer_Adam.step()

            self.scheduler_adam.step(loss)

            after_lr = self.optimizer_Adam.param_groups[0]["lr"]
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}, LR: {after_lr}')

        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        c = self.net_c(x, t)
        q = self.net_q(x, t)
        f_c, f_q = self.net_f(x, t)

        c = c.detach().cpu().numpy()
        q = q.detach().cpu().numpy()
        f_c = f_c.detach().cpu().numpy()
        f_q = f_q.detach().cpu().numpy()

        return c, q, f_c, f_q

