import numpy as np
import scipy
from matplotlib import pyplot as plt
import scipy
from torch.utils.data import DataLoader, TensorDataset
from pinns.pinns_burger_v03 import PhysicsInformedNN
import torch

# Set problem parameters
nu = 0.01/np.pi
N_u = 2000
layers = [2, 40, 40, 40, 1]

# source: https://github.com/maziarraissi/PINNs
data = scipy.io.loadmat('pinns/data/burgers_shock.mat')

# extract domain
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x, t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

noise = 0

# Split the dataset into training and validation sets
idx = np.random.choice(X_star.shape[0], N_u, replace=False)

X_u_train = X_star[idx,:]
u_train = u_star[idx,:]

X_u_val = X_star[~idx,:]
u_val = u_star[~idx,:]
# training

model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
val_dataset = TensorDataset(torch.tensor(X_u_val, dtype=torch.float32), torch.tensor(u_val, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
model.train(5000)

ypred, fpred = model.predict(X_star)

plt.plot(u_star, ypred, 'o', linestyle='')
plt.show()

ypred, fpred = model.predict(X_star)

plt.plot(u_star, ypred, 'o', linestyle='', color='orange')
plt.title('True y vs ypred0')
plt.show()

# Additional diagnostics plot
plt.plot(X_star[:, 0], u_star, 'o', linestyle='')
plt.show()

plt.plot(X_star[:, 0], ypred, 'o', linestyle='')
plt.show()

plt.plot(X_star[:, 1], u_star, 'o', linestyle='')
plt.show()


plt.plot(X_star[:, 1], ypred, 'o', linestyle='')
plt.show()