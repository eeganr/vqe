import torch
from copy import deepcopy


backend = 'torch'

if backend == 'torch':
    import torch as bknd
    bknd_tensor = bknd.tensor
    bknd_array = bknd_tensor
elif backend == 'numpy':
    import numpy as bknd
    bknd_array = bknd.array
    bknd_tensor = bknd_array

sigma_eye = bknd_tensor([[1., 0.],
                          [0., 1.]])
sigmaI = sigma_I = sigma0 = sigma_0 = sigmaeye = sigma_eye

sigma_x = bknd_tensor([[0., 1.],
                        [1., 0.]])
sigma1 = sigma_1 = sigmax = sigma_x

sigma_y = bknd_tensor([[0., -1.j],
                        [1.j, 0.]])
sigma2 = sigma_2 = sigmay = sigma_y

sigma_z = bknd_tensor([[1., 0.],
                        [0., -1.]])
sigma3 = sigma_3 = sigmaz = sigma_z

sigma_plus = bknd_tensor([[0., 1.],
                           [0., 0.]])
sigmap = sigma_p = sigma_plus

sigma_minus = bknd_tensor([[0., 0.],
                            [1., 0.]])
sigmam = sigma_m = sigma_minus

Sx = S_x = sigma_x / 2
Sy = S_y = sigma_y / 2
Sz = S_z = sigma_z / 2

CX = bknd_tensor([[1., 0., 0., 0.], # TODO: Check is this is correct
                  [0., 1., 0., 0.], 
                  [0., 0., 0., 1.], 
                  [0., 0., 1., 0.]])

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
    print(psi_out, gate, (list(range(-num_sites, 0)), 
                                            list(range(-num_sites, 0))))
    print('############')
    psi_out = bknd.tensordot(psi_out, gate, (list(range(-num_sites, 0)), 
                                            list(range(-num_sites, 0))))
    # move the contracted dimensions back
    for site in reversed(sites):
        if site < 0:
            site = nb_qbits + site
        psi_out = psi_out.swapaxes(site, -1).squeeze(-1)
    return psi_out# .contiguous()


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
        probs, psis_out = measure_remove_single_site(psi_in, site, basis, check_valid, eps)
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
    

def normalize(psi):
    """
    psi: wave function of shape (d,d,d,d,d...)\
    """
    psi = psi / bknd.linalg.norm(psi.ravel())

    return psi


def multikron(*tensors):
    """
    kronecker product (kron) for multiple tensors
    """
    if type(tensors[0]) is list or type(tensors[0]) is tuple:
        tensors = tensors[0]
    result = tensors[0]
    for tensor in tensors[1:]:
        result = bknd.kron(result, tensor)
    return result


def one_body_gate(gate, loc, num_qbits, default_gate=sigmaeye, device=None):
    """
    loc: the location of the gate
    num_qbits: the total number of qubits
    default_gate: the gate to be placed on the rest sites, default to be identity
    """
    if device is not None:
        default_gate = default_gate.to(device)
    result = [default_gate] * num_qbits
    assert gate.shape == (2, 2)
    assert 0 <= loc < num_qbits
    result[loc] = gate
    return multikron(*result)


def many_body_gate(gates, locs, num_qbits, default_gate=sigmaeye, device=None):
    """
    locs: the locations of each gate
    num_qbits: the total number of qubits
    default_gate: the gate to be placed on the rest sites, default to be identity
    """
    if device is not None:
        default_gate = default_gate.to(device)
    result = [default_gate] * num_qbits
    for gate, loc in zip(gates, locs):
        assert gate.shape == (2, 2)
        assert 0 <= loc < num_qbits
        result[loc] = gate
    return multikron(*result)


def gen_tfim_ham(h, nb_qbits, periodic=False, device=None):
    """
    generate the full transverse file ising model Hamiltonian
    """
    ham = bknd.zeros((2 ** nb_qbits, 2 ** nb_qbits))
    if device is not None:
        ham = ham.to(device)
        sigmax_d = sigmax.to(device)
        sigmaz_d = sigmaz.to(device)
    else:
        sigmax_d = sigmax
        sigmaz_d = sigmaz
    for i in range(nb_qbits):
        ham += -one_body_gate(sigmax_d, i, nb_qbits, device=device) * h # negative sign is convention
    for i in range(nb_qbits-1):
        ham += -many_body_gate([sigmaz_d, sigmaz_d], [i, i+1], nb_qbits, device=device) # negative sign is convention
    if periodic:
        ham += -many_body_gate([sigmaz_d, sigmaz_d], [nb_qbits-1, 0], nb_qbits, device=device) # negative sign is convention
    return ham


def expect_value(operator, psi, normalize=False):
    """
    computes the expectation value of <psi|H|psi>
    operator: an operator can be either a matrix or a function that computes Opsi
    psi: needs to have shape compatible with operator
    normalize: whether to explicitly normalize
    return <psi|O|psi> if not normalize else <psi|O|psi>/<psi|psi>
    """
    if callable(operator):
        Opsi = operator(psi).ravel()
        psi = psi.ravel()
        ev = bknd.vdot(psi @ Opsi)
    else:
        psi = psi.ravel()
        ev = psi.conj() @ operator @ psi
    if normalize:
        normsq = bknd.vdot(psi, psi)
        ev /= normsq
    return ev

def Rx_gate(theta):
    return bknd_tensor([[bknd.cos(theta/2), -1j*bknd.sin(theta/2)], [-1j*bknd.sin(theta/2), bknd.cos(theta/2)]])


def Ry_gate(theta):
    return bknd_tensor([[bknd.cos(theta/2), -bknd.sin(theta/2)], [bknd.sin(theta/2), bknd.cos(theta/2)]])


def Rz_gate(theta):
    return bknd_tensor([[bknd.exp(-1j*theta/2), 0], [0, bknd.exp(1j*theta/2)]])


def su2_transform_psi(psi0, thetas):
    qubits = thetas.shape[0]
    gates = thetas.shape[1]

    psi_out = psi0.detach().clone().to(bknd.complex64)

    # Check if psi0 fits number of qubits
    assert qubits == len(psi0.shape)
    assert gates % 2 == 0

    for col in range(gates):
        for row in range(qubits):
            if col % 2 == 0:
                gate = Ry_gate(thetas[row][col])
                psi_out = apply_many_body_gate(psi_out, gate.to(bknd.complex64), qubits, [row])
            else: 
                gate = Rz_gate(thetas[row][col])
                psi_out = apply_many_body_gate(psi_out, gate.to(bknd.complex64), qubits, [row])
                if col < gates - 1 and row == qubits - 1:
                    for row_CX in reversed(range(qubits - 1)):
                        psi_out = apply_many_body_gate(psi_out, CX.to(bknd.complex64), qubits, [row_CX, row_CX+1])
                        # TODO: Problem here? Changes shape of psi?
    
    return psi_out
    
def su2_energy_from_thetas(psi0, thetas):
    psi_out = su2_transform_psi(psi0, thetas)
    ham = gen_tfim_ham(0, thetas.shape[0])
    return expect_value(ham, psi_out, True)
# Neural network will create a bunch of gates
# Different probability for each gate
# Use some search algorithm for gate creation?
# Reward = energy of the system (some relation)
# |psi> ---circ---> |psi'>
# tune theta parameters to maximize reward

# Simulated annealing for thetas? - no feedback?
# Use MCTS? 
# Train policy network independently
