from quantumsimulator import QuantumSimulator as qs
import torch as bknd

qubits = 3

psi = qs.normalize(bknd.rand((2,)*qubits))

ham = qs.gen_tfim_ham(1, qubits)

thetas = bknd.zeros(qubits, 6)



# Input all 0 psi state

# Try out image model, adapt

# Thetas are just 2d array, similar to image

# Reward = energy

# Measurement: print(psi.conj() @ ham @ psi)