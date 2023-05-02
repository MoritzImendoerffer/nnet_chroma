"""
Merging PINN and INVNN

Implementing the INVNN using FrEIA

because INN has also 2 outputdims, I tried to optimize so that both output vectors stay the same
This approach is very slow and probably a fail
"""

import torch  # arrays on GPU
import numpy as np
import scipy
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from collections import OrderedDict

torch.manual_seed(0)
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


def subnet_fc(dims_in, dims_out):
    """Return a feed-forward subnetwork, to be used in the coupling blocks below"""
    return torch.nn.Sequential(torch.nn.Linear(dims_in, 128), torch.nn.Tanh(),
                               torch.nn.Linear(128, 128), torch.nn.Tanh(),
                               torch.nn.Linear(128, dims_out))


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self, X, u, lb, ub):

        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        # settings
        self.lambda_1 = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True).to(device))
        self.lambda_2 = torch.nn.Parameter(torch.tensor([-6.0], requires_grad=True).to(device))

        # construct the INN (not containing any operations so far)
        input_dims = (2, )
        inn = Ff.SequenceINN(*input_dims)

        # append coupling blocks to the sequence of operations
        for k in range(8):
            inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc)

        self.dnn = inn.to(device)

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
            #line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters()
        )
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_u(self, x, t):
        x_, u = self.dnn(torch.cat([x, t], dim=1))
        return x_

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        u, u1 = self.net_u(x, t).T
        u = u[:, None]
        u1 = u1[:, None]

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

        u1_t = torch.autograd.grad(
            u1, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u1_x = torch.autograd.grad(
            u1, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u1_xx = torch.autograd.grad(
            u1_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx + u1_t + lambda_1 * u1 * u1_x - lambda_2 * u1_xx
        return f

    def loss_func(self):
        """For local optimization using BFGS"""
        u_pred = self.net_u(self.x, self.t)
        u_pred, u1_pred = u_pred.T
        u1_pred = u1_pred[:, None]
        u_pred = u_pred[:, None]

        f_pred = self.net_f(self.x, self.t)
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean((self.u - u1_pred) ** 2) + torch.mean(f_pred ** 2)
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
        self.dnn.train()
        for epoch in range(nIter):
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            loss.backward()
            self.optimizer_Adam.step()

            if epoch % 100 == 0:
                print(
                    'B: It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
                    (
                        epoch,
                        loss.item(),
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


nu = 0.01 / np.pi

N_u = 2000


data = scipy.io.loadmat('data/burgers_shock.mat')

t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

noise = 1e-5

# create training set
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx, :]
u_train = u_star[idx, :]

# training
model = PhysicsInformedNN(X_u_train, u_train, lb, ub)
model.train(10000)
model.predict(X_u_train)
