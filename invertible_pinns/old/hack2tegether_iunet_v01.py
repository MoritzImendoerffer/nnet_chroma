"""
Merging PINN and INVNN

Implementing mit IUNET brauch conda env iunet

Weiß nicht, ob geeignet inchannels nur als int:
        model = iUNet(
            in_channels=64,
            dim=1,
            architecture=(2, 2, 2, 2, 2)
        )
basiert auf memcnn, auch mit konstanten channels:
https://github.com/silvandeleemput/memcnn
"""

import torch # arrays on GPU
from iunets import iUNet
import numpy as np
from collections import OrderedDict
import scipy
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
        model = iUNet(
            in_channels=64,
            dim=1,
            architecture=(2, 2, 2, 2, 2)
        )
        model.print_layout()
        self.dnn = model.to('cuda')

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
        return u[:,None]

    def net_f(self, x, t):
        """ The pytorch autograd version of calculating residual """
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
        """For local optimization using BFGS"""
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
        # not sure why the following line is needed
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
