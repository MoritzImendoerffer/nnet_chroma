"""
Merging PINN and INVNN

Implementing the INVNN using FrEIA

y[:, 0] for estimation of u
y[:, 1] for estimation of f
seems to work better -> Nope Overfitting seems to occur
"""

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import scipy
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from collections import OrderedDict
from matplotlib import pyplot as plt

#torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

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
            inn.append(Fm.AllInOneBlock, subnet_constructor=self.subnet_fc,
                       affine_clamping=3.0,
                       gin_block=True,
                       global_affine_init=1.0,
                       global_affine_type='SOFTPLUS',
                       permute_soft=True,
                       learned_householder_permutation=0,
                       reverse_permutation=False)
            # slower but better at version 0.4
            # clamp 3 scheint besser zu funktionieren
            # inn.append(Fm.RNVPCouplingBlock, subnet_constructor=self.subnet_fc, clamp=3)
            # inn.append(Fm.RNVPCouplingBlock, subnet_constructor=self.subnet_fc, clamp=2)


        self.dnn = inn.to(device)

        self.dnn.register_parameter('lambda_1', self.lambda_1)
        self.dnn.register_parameter('lambda_2', self.lambda_2)
        lr = 1e-3
        l2_reg = 2e-5
        trainable_parameters = [p for p in self.dnn.parameters() if p.requires_grad]

        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            trainable_parameters,
            lr=1,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        # self.optimizer = torch.optim.SGD(
        #     self.dnn.parameters(),
        #     lr=0.05,
        #     momentum=0.9
        # )
        # gamma = decaying factor
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=0, verbose=True)

        self.optimizer_Adam = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=l2_reg)
        # self.optimizer_Adam_1 = torch.optim.Adam(self.dnn.parameters(), lr=0.001)

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
        uhats = self.net_u(x, t)
        # This is a source for error because the indices are hardcoded in two different functions
        # y values
        u = uhats[:, 1]
        u = u[:, None]

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

    def net_u_inverse(self, y):
        x, log_jac_det_inv = self.dnn(y, rev=True)
        return x

    def mse_loss_u(self):
        # training so that the output 2 is equal the time
        uhats = self.net_u(self.x, self.t)

        # y values
        u_pred = uhats[:, 0]
        u_pred = u_pred[:, None]

        loss = torch.mean((self.u - u_pred) ** 2)
        return loss

    def inverse_loss(self):
        # training so that the output 2 is equal the time
        uhats = self.net_u(self.x, self.t)

        xhat = self.net_u_inverse(uhats)

        # y values
        loss = torch.mean((self.x - xhat[:, 0]) ** 2) + torch.mean((torch.zeros_like(self.x) - xhat[:, 1]) ** 2)
        return loss

    def mse_loss_f(self):
        # training so that the output 2 is equal the time
        f_pred = self.net_f(self.x, self.t)

        loss = torch.mean(f_pred ** 2)
        return loss

    def mse_residual(self):
        uhats = self.net_u(self.x, self.t)

        # y values
        u_pred = uhats[:, 1]
        u_pred = u_pred[:, None]

        loss = torch.mean((self.u - u_pred) ** 2)
        return loss

    def loss_func(self):
        """For local optimization using BFGS"""

        loss_u = self.mse_loss_u()
        loss_f = self.mse_loss_f()

        loss_inv = self.inverse_loss()

        loss = loss_u + loss_f + loss_inv
        # loss = loss_u + loss_f + loss_r
        self.optimizer.zero_grad()
        # loss_u.backward(retain_graph=True)
        # loss_f.backward(retain_graph=True)
        loss.backward(retain_graph=True)

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

            loss_inv = self.inverse_loss()

            loss = loss_u + loss_f + loss_inv

            # loss = loss_u + loss_f + loss_r
            # Backward and optimize
            self.optimizer_Adam.zero_grad()
            #self.optimizer_Adam_1.zero_grad()
            # loss_u.backward(retain_graph=True)
            # loss_f.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            self.optimizer_Adam.step()
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

nu = 0.01 / np.pi
N_u = 2000
data = scipy.io.loadmat('data/burgers_shock.mat')
t = data['t'].flatten()[:, None]
# sd = 1e-5
# noise = scipy.stats.norm.rvs(loc=0, scale=sd, size=len(t))[:, None]
#
# t[t != 0] += noise[t != 0]
# t[t < 0] = 0

x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]
                    ))
u_star = Exact.flatten()[:, None]
# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

# create training set
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx, :]
u_train = u_star[idx, :]

c = (4, 11)
c = (5, 6)
#c = (512, 5)
model = PhysicsInformedNN(X_u_train, u_train, lb, ub, n_sub=c[0], n_node=c[1])
model.train(1000)

# Run 1: Winner: (4, 10), Run2: Winner: Winner: (5, 6)

ypred, fpred = model.predict(X_star)

plt.plot(u_star, ypred[:, 0], 'o', linestyle='', color='orange')
plt.title('True y vs ypred0')
plt.show()

ypred, fpred = model.predict(X_u_train)

plt.plot(u_train, ypred[:, 0], 'o', linestyle='', color='orange')
plt.title('True y vs ypred0')
plt.show()

plt.plot(u_train, ypred[:, 1], 'o', linestyle='', color='orange')
plt.title('True y vs ypred1')
plt.show()

plt.plot(ypred[:, 0], ypred[:, 1], 'o', linestyle='', color='C3')
plt.title('ypred0 vs ypred1')
plt.show()

plt.plot(X_u_train[:, 0], ypred[:, 0], 'o', linestyle='')
plt.title('True X0 vs ypred0')
plt.show()

plt.plot(X_u_train[:, 1], ypred[:, 0], 'o', linestyle='')
plt.title('True X1 vs ypred0')
plt.show()

plt.plot(X_u_train[:, 0], ypred[:, 1], 'o', linestyle='', color='gray')
plt.title('True X0 vs ypred1')
plt.show()

plt.plot(X_u_train[:, 1], ypred[:, 1], 'o', linestyle='', color='gray')
plt.title('True X1 vs ypred1')
plt.show()

plt.plot(X_u_train[:, 0], u_train, 'o', linestyle='', color='green')
plt.title('True X0 vs y')
plt.show()


plt.plot(X_u_train[:, 1], u_train, 'o', linestyle='', color='green')
plt.title('True X1 vs y')
plt.show()
print('done')