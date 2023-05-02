import numpy as np
import scipy
from matplotlib import pyplot as plt
import scipy
from torch.utils.data import DataLoader, TensorDataset
from invertible_pinns.inv_pinns_freia_v06 import PhysicsInformedNN
import torch

nu = 0.01 / np.pi

N_u = 2000


data = scipy.io.loadmat('invertible_pinns/data/burgers_shock.mat')

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

'''
This part defines the architecture of the building blocks for the invertible NN
TODO: understand the implications better.
'''
# bisher nsub = 20, n_node = 4 oder 5 als Gewinner, 60/4 war auch im Bereich von 1e-5, auch (6, 5), (6, 6) und (6, 10)
n_sub = [3, 4, 5, 6]
n_node = [3, 4, 5, 6, 10]

combs = []
for ns in n_sub:
    for no in n_node:
        combs.append((ns, no))


# training
models = []
for c in combs:
    print(20*'=')
    print(c)
    print(20 * '=')
    model = PhysicsInformedNN(X_u_train, u_train, lb, ub, n_sub=c[0], n_node=c[1])
    model.train(10000)
    models.append(model)
# Run 1: Winner: (4, 10), Run2: Winner: Winner: (5, 6)

print('Done')
losses = [m.optimizer.state_dict()['state'][0]['prev_loss'] for m in models]
min_idx = np.argmin(losses)

print(f'Winner: {combs[min_idx]}')
best_model = models[min_idx]
ypred, fpred = best_model.predict(X_u_train)

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