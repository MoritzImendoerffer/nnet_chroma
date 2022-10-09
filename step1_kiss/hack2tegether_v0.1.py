"""
Merging PINN and INVNN
"""

import torch # arrays on GPU
import torch.autograd as autograd #build a computational graph
import torch.nn as nn ## neural net library
import torch.nn.functional as F ## most non-linearities are here
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class NN(nn.Module):

    def __int__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.nn_simple = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)

        )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.nn_simple(x)
            return logits
model = NN().to(device)
