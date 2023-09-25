import numpy as np
from copy import deepcopy
def apply_many_body_gate(psi_in, gate, nb_qbits, sites):
    """
    psi_in: input wave function of shape (d,d,d,d,d...)
    gate: gate of shape (d,d,..., d,d,...) to be applied on psi_in. 
    nb_qbits: number of qubits of psi_in
    sites: the sites to apply the gate
    """
    num_sites = len(sites)
    # assert num_sites * 2 == gate.ndim, f''
    # will not change psi_in anyway
    psi_out = psi_in
    # move the contracted dimensions to last for easier manipulation
    for site in sites:
        if site < 0:
            site = nb_qbits + site
        psi_out = psi_out.reshape(*psi_out.shape, 1).swapaxes(site, -1)
    psi_out = np.tensordot(psi_out, gate, (list(range(-num_sites, 0)), 
                                              list(range(-num_sites, 0))))
    # move the contracted dimensions back
    for site in reversed(sites):
        if site < 0:
            site = nb_qbits + site
        psi_out = psi_out.swapaxes(site, -1).squeeze(-1)
    return psi_out# .contiguous()

def measure(psi_in, qubit, basis=[0, 1]):
    
    dims = len(psi_in.shape)
    meas_outcomes = psi_in.shape[qubit]

    slices = [""] * meas_outcomes
    for i in range(dims):
        if i == qubit:
            for j in range(meas_outcomes):
                slices[j] += f"{j}, "
        else:
            for j in range(meas_outcomes):
                slices[j] += ":, "
    for slice in slices: 
        slice = slice[:-2]

    psis = []
    for i in range(meas_outcomes):
        psis.append(eval("psi_in[" + slices[i] + "]"))

    probs = [np.linalg.norm(psi) ** 2 for psi in psis]

    psis = [psi / np.linalg.norm(psi) for psi in psis]

    return probs, psis
    

psi = np.ones((3, 3, 3, 3))
psi = np.random.randn(3, 3, 3, 3, 3)

psi = psi / np.linalg.norm(psi.ravel())

"""
gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
gate = gate.reshape(2, 2, 2, 2)

psiprime = apply_many_body_gate(psi, gate, 4, [0, 2])
"""

probs, psis = measure(psi, 1)

print(probs)



# print(psiprime)

# Put in another qubit

# psi0 = np.tensordot(psi0, np.array([1, 0]), 0)
