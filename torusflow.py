import matplotlib.pyplot as plt
import torch
import zuko
from quantumsimulator import *

qubits = 3

psi = bknd.zeros((2,)*qubits)

psi[(0,)*qubits] = 1

psi = normalize(psi)

thetas = bknd.rand((2, qubits, 4))

x = thetas.reshape(thetas.shape[0],thetas.shape[1]*thetas.shape[2])

y = x.reshape(x.shape[0], thetas.shape[1], thetas.shape[2])

ham = gen_tfim_ham(0.5, thetas.shape[1])+0j

def log_energy(x):
    angles = x.reshape(x.shape[0], thetas.shape[1], thetas.shape[2])
    energies = su2_energy_from_thetas_batched(psi, ham, angles.squeeze(1)).real
    return energies

#x1 = torch.linspace(-3, 3, 64)
#x2 = torch.linspace(-3, 3, 64)

#x = torch.stack(torch.meshgrid(x1, x2, indexing='xy'), dim=-1)

#energy = log_energy(x).exp()

flow = zuko.flows.NCSF(features=thetas.shape[1]*thetas.shape[2], transforms=3, hidden_features=(64, 64))
flow = zuko.flows.Flow(flow.transform.inv, flow.base)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

for epoch in range(8):
    losses = []

    for _ in range(256):
        x, log_prob = flow().rsample_and_log_prob((256,)) 

        energies = log_energy(x)
        if _ % 50 == 0: print(_, energies.mean(), energies.min())
        loss = (energies - energies.mean()) * log_prob
        loss = loss.mean()
        # loss = log_prob.mean() - log_energy(x).mean() <- this was the original
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())

    losses = torch.stack(losses)

    print(f'({epoch})', losses.mean().item(), '±', losses.std().item())

