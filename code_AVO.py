
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self, n_features=1):
        super(SimpleModel, self).__init__()
        self.lin1 = nn.Linear(n_features, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 1)

    def forward(self, x):
       x = F.relu(self.lin1(x))
       return self.lin3(x)


def make_optimizer_step(loss, optimizer, retain_graph=None):
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
    optimizer.step()


def get_gradient_penalty(real_data, fake_data, critic):
    eps = torch.distributions.uniform.Uniform(0., 1.).sample(real_data.shape)

    interpolated_data = eps*real_data + (1.-eps)*fake_data
    interpolated_data = torch.autograd.Variable(interpolated_data, requires_grad=True)


    critic_iterpolated = critic(interpolated_data)

    gradient = torch.autograd.grad(outputs=critic_iterpolated, inputs=interpolated_data,
                    grad_outputs=torch.ones(critic_iterpolated.size()),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient = gradient.view(gradient.size(0), -1)

    gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
