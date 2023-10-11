import numpy as bknd
import numpy as np
import torch
from copy import deepcopy


class QuantumSimulator:
    def __init__(self):
        return
    
    @staticmethod
    def apply_many_body_gate(psi_in, gate, nb_qbits, sites):
        """
        psi_in: ibkndut wave function of shape (d,d,d,d,d...)
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
        psi_out = bknd.tensordot(psi_out, gate, (list(range(-num_sites, 0)), 
                                                list(range(-num_sites, 0))))
        # move the contracted dimensions back
        for site in reversed(sites):
            if site < 0:
                site = nb_qbits + site
            psi_out = psi_out.swapaxes(site, -1).squeeze(-1)
        return psi_out# .contiguous()

    @staticmethod
    def measure_remove_single_site(psi_in, site, basis=None, check_valid=False, eps=1e-14):
        """
        psi_in: ibkndut wave function of shape (d,d,d,d,d...)
        site: the site to measure
        basis: measurement basis if not provided, will assume to be the z (computational) basis
            if provided, it should be a matrix, with each row a basis vector (not conjugated). the basis vectors should form an orthonomal basis
        check_valid: checks if the measurement basis is a valid basis set
        return: a list of output wave functions (will be combined into an array with one more dimension) with the measured qubit removed, and the corrsponding probabilities
        """
        if basis is not None:
            if check_valid:
                assert bknd.allclose(basis.T.conj() @ basis, bknd.eye(basis.shape[-1]))
            # apply gate to site
            psis_out = bknd.tensordot(basis.conj(), psi_in, axes=(1, site))
        else:
            # move site to front
            psis_out = bknd.moveaxis(psi_in, tuple(range(0, site+1)), (tuple(range(1, site+1)) + (0,)))
        norms = bknd.linalg.norm(psis_out.reshape(psis_out.shape[0], -1), 2, axis=1)
        probs = norms**2
        psis_out = psis_out / norms.reshape(-1, *([1]*len(psis_out.shape[1:]))).clip(eps, None)
        return probs, psis_out

    @staticmethod
    def measure_single_site(psi_in, site, basis=None, check_valid=False, eps=1e-14):
        """
        psi_in: ibkndut wave function of shape (d,d,d,d,d...)
        site: the site to measure
        basis: measurement basis if not provided, will assume to be the z (computational) basis
            if provided, can be in one of the following forms：
                1. a matrix, with each row a basis vector (not conjugated). the basis vectors should form an orthonomal basis
                2. a 3D array, containing matrices of measurement projectors. the measurement projectors should be positive semidefinite (we will not check this), and sum up to identity matrix
        check_valid: checks if the measurement basis is a valid basis set, will not check for positive semidefinite.
        return: a list of output wave functions (will be combined into an array with one more dimension), and the corrsponding probabilities
        """
        
        # if basis is form 1
        if basis is None or len(basis.shape) == 2:
            probs, psis_out = QuantumSimulator.measure_remove_single_site(psi_in, site, basis, check_valid, eps)
            # put back the removed qubit
            psis_out_shape = psis_out.shape
            # reshape the wave function
            psis_out = psis_out.reshape(psis_out_shape[0], 1, -1)
            if basis is None:
                basis = bknd.eye(psis_out.shape[0])
                if bknd == torch:
                    basis = basis.to(psis_out.device)
            basis = basis[:, :, None]
            # batch matrix multiplication over the dim with shape 1, which effectively prepend the qubit to the begining
            psis_out = bknd.matmul(basis, psis_out)
            # reshape psi back
            psis_out = psis_out.reshape(psis_out_shape[0], -1, *psis_out_shape[1:])
            # permute the 0th qubit back to its position
            psis_out = bknd.moveaxis(psis_out, tuple(range(1, site+2)), (site+1,) + tuple(range(1, site+1))) # +1 because of the batch dimension as the 0th dimension
            return probs, psis_out
            
        # if basis is form 2
        elif len(basis.shape) == 3:
            if check_valid:
                # only check if sums up to identity
                assert bknd.allclose(basis.sum(0), bknd.eye(basis.shape[-1]))
            # apply the basis in a batch form and move the axis back to the correct sites
            psis_out = bknd.moveaxis(bknd.tensordot(basis, psi_in, axes=(2, site)), tuple(range(1, site+2)), (site+1,) + tuple(range(1, site+1))) # +1 because of the batch dimension as the 0th dimension
            norms = bknd.linalg.norm(psis_out.reshape(psis_out.shape[0], -1), 2, axis=1)
            probs = norms**2
            psis_out = psis_out / norms.reshape(-1, *([1]*len(psis_out.shape[1:]))).clip(eps, None)
            return probs, psis_out
        
    @staticmethod
    def normalize(psi):
        """
        psi: wave function of shape (d,d,d,d,d...)\
        """
        psi = psi / bknd.linalg.norm(psi.ravel())

        return psi
    

# Neural network will create a bunch of gates
# Different probability for each gate
# Use some search algorithm for gate creation?
# Reward = energy of the system (some relation)
# |psi> ---circ---> |psi'>
# tune theta parameters to maximize reward

# Simulated annealing for thetas? - no feedback?
# Use MCTS? 
# Train policy network independently
