import numpy as np
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

G = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
E = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6"), ("6", "7"), 
     ("8", "9"), ("9", "10"), ("10", "11"), ("11", "12"), ("12", "13"), ("13", "14"),
     ("1", "8"), ("2", "9"), ("3", "10"), ("4", "11"), ("5", "12"), ("6", "13"), ("7", "14")]
