import matplotlib.pyplot as plt
import torch
import zuko
from quantumsimulator import *
import math
import numpy

qubits = 2

psi = bknd.zeros((2,)*qubits)

psi[(0,)*qubits] = 1

psi = normalize(psi)

thetas = bknd.zeros((1, qubits, 1))

x = thetas.reshape(thetas.shape[0],thetas.shape[1]*thetas.shape[2])

y = x.reshape(x.shape[0], thetas.shape[1], thetas.shape[2])

ham = gen_tfim_ham(0, thetas.shape[1])+0j

def energy(x):
    angles = x.reshape(x.shape[0], thetas.shape[1], thetas.shape[2])
    energies = su2_energy_from_thetas_batched(psi, ham, angles.squeeze(1)).real
    return energies

#x1 = torch.linspace(-3, 3, 64)
#x2 = torch.linspace(-3, 3, 64)

#x = torch.stack(torch.meshgrid(x1, x2, indexing='xy'), dim=-1)

#energy = log_energy(x).exp()


flow = zuko.flows.NCSF(features=thetas.shape[1]*thetas.shape[2], transforms=3, hidden_features=(8, 8))
flow = zuko.flows.Flow(flow.transform.inv, flow.base)


optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

for epoch in range(8):
    losses = []

    for _ in range(256):
        x, log_prob = flow().rsample_and_log_prob((256,)) 

        energies = energy(x.detach())
        if _ % 50 == 0: 
            x1 = numpy.linspace(-8*math.pi, 8*math.pi, 1024, endpoint=False)
            x2 = numpy.linspace(-8*math.pi, 8*math.pi, 1024, endpoint=False)

            x = torch.stack(torch.meshgrid(torch.tensor(x1).float(), torch.tensor(x2).float(), indexing='xy'), dim=-1)

            x1 = numpy.linspace(-8*math.pi, 8*math.pi, 1024, endpoint=False) + 2 * math.pi
            x2 = numpy.linspace(-8*math.pi, 8*math.pi, 1024, endpoint=False) + 2 * math.pi

            x_prime = torch.stack(torch.meshgrid(torch.tensor(x1).float(), torch.tensor(x2).float(), indexing='xy'), dim=-1)

            prob = flow().log_prob(x).exp()

            prob_prime = flow().log_prob(x_prime).exp()

            prob -= prob_prime

            print(prob.shape)

            plt.figure(figsize=(4.8, 4.8))
            plt.imshow(prob.detach())
            plt.show()
            print(_, energies.mean(), energies.min())
        # if _ % 50 == 0: print(x[1])
        loss = (energies - energies.mean()) * log_prob
        loss = loss.mean()
        # loss = log_prob.mean() - log_energy(x).mean() <- this was the original
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach())
        

    losses = torch.stack(losses)

    print(f'({epoch})', losses.mean().item(), '±', losses.std().item())

