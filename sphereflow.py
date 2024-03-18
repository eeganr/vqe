import matplotlib.pyplot as plt
import torch
import zuko
from quantumsimulator import *

def log_energy(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.sin(torch.pi * x1) - 2 * (x1 ** 2 + x2 ** 2 - 2) ** 2

#x1 = torch.linspace(-3, 3, 64)
#x2 = torch.linspace(-3, 3, 64)

#x = torch.stack(torch.meshgrid(x1, x2, indexing='xy'), dim=-1)

#energy = log_energy(x).exp()

flow = zuko.flows.NCSF(features=8, transforms=3, hidden_features=(64, 64))
flow = zuko.flows.Flow(flow.transform.inv, flow.base)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

for epoch in range(8):
    losses = []

    for _ in range(256):
        x, log_prob = flow().rsample_and_log_prob((256,))  # faster than rsample + log_prob
        print(x.shape)
        loss = log_prob.mean() - log_energy(x).mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())

    losses = torch.stack(losses)

    print(f'({epoch})', losses.mean().item(), '±', losses.std().item())

