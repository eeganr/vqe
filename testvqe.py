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

psi = np.ones((2, 2, 2, 2))

psi = psi / np.linalg.norm(psi.ravel())


gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
gate = gate.reshape(2, 2, 2, 2)

psiprime = apply_many_body_gate(psi, gate, 4, [0, 2])

print(psiprime)

psi0 = psiprime[:, 0, :, :]

psi1 = psiprime[:, 1, :, :]

p0 = np.linalg.norm(psi0)

p1 = np.linalg.norm(psi1)

psi0 /= p0
psi1 /= p1

p0 = p0 ** 2
p1 = p1 ** 2

print(p0, p1, psi0, psi1)

# Put in another qubit

psi0 = np.tensordot(psi0, np.array([1, 0]), 0)

print(psi0)