import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Constants
v = 0.5  # Example value
omega = 0.1  # Example value
g = 1  # Example value
hbar = 1  # Planck's constant (set to 1 for simplicity)

# Parameters for simulation
N = 10  # Size of the Hilbert space for the oscillator
dt = 0.01  # Small time step
total_time = 80  # Total time for simulation
n_steps = int(total_time / dt)

# Operators
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_y = np.array([[0, -1j], [1j, 0]])


def a_operator(N):
    """Creation and annihilation operators for N-dimensional Hilbert space."""
    a = np.zeros((N, N), dtype=complex)
    for n in range(N-1):
        a[n, n + 1] = np.sqrt(n+1)
    return a

def u(h, t):
    return la.expm(-1j * h * t / hbar)


def operators(N):
    """First time-dependent Hamiltonian."""
    a = a_operator(N)
    a_dagger = a.T.conj()
    hs = v * np.kron(sigma_x, np.eye(N))
    hb = omega * np.kron(np.eye(2), np.dot(a_dagger, a))
    hsb = g * np.kron(sigma_z, (a_dagger + a))
    hsb2 = g * np.kron(sigma_z, (a - a_dagger))
    h = hs + hb + hsb
    h_transform = lambda t: g/omega * np.kron(sigma_z, 
                                         a * (np.exp(1j*omega*t)-1) - a.T * (np.exp(-1j*omega*t)-1)
                                         )
    return h, hs, hb, hsb, a, a_dagger, h_transform


def von_neumann_entropy(rho):
    """Calculate the von Neumann entropy of a density matrix."""
    eigenvalues = la.eigvalsh(rho)
    return -sum(eigenval * np.log(eigenval) for eigenval in eigenvalues if eigenval > 0)


def occupation_number(state, N):
    a = a_operator(N)
    a_dagger = a.T.conj()
    number_op = np.dot(a_dagger, a)
    number_op_full = np.kron(np.eye(2), number_op)  # Extend to full Hilbert space
    return np.real(np.vdot(state, number_op_full @ state))


# Initial state (example: spin up, oscillator in ground state)
psi0 = np.kron(np.array([1, 0]), np.array([1] + [0] * (N - 1)))

# Time evolution for both Hamiltonians
sz_expect_1 = []
sz_expect_2 = []
entanglement_1 = []
entanglement_2 = []
occupation_numbers_1 = []
occupation_numbers_2 = []
psi_t_1 = psi0
psi_t_2 = psi0

h, hs, hb, hsb, a, a_dagger, h_transform = operators(N)
h_test = g * np.kron(sigma_z, a_dagger+a)
for step in range(n_steps):
    t = step * dt
    psi_t_1 = u(-g/omega * np.kron(sigma_z, (a-a_dagger)), 1j) @ u(h, t)@psi0
    psi_t_2 = u(h_transform(t), 1j) @u(h, t)@psi0
    sz_expect_1.append(np.real(np.vdot(psi_t_1, np.kron(sigma_z, np.eye(N)) @ psi_t_1)))
    sz_expect_2.append(np.real(np.vdot(psi_t_2, np.kron(sigma_z, np.eye(N)) @ psi_t_2)))

    rho_1 = np.outer(psi_t_1, psi_t_1.conj())
    rho_2 = np.outer(psi_t_2, psi_t_2.conj())
    reduced_rho_1 = np.trace(rho_1.reshape(N, 2, N, 2), axis1=1, axis2=3)
    reduced_rho_2 = np.trace(rho_2.reshape(N, 2, N, 2), axis1=1, axis2=3)
    entanglement_1.append(von_neumann_entropy(reduced_rho_1))
    entanglement_2.append(von_neumann_entropy(reduced_rho_2))

    occupation_numbers_1.append(occupation_number(psi_t_1, N))
    occupation_numbers_2.append(occupation_number(psi_t_2, N))

# Plotting the results
times = np.linspace(0, total_time, n_steps)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Spin population plot
axs[0, 0].plot(times, sz_expect_1, label='new')
axs[0, 0].plot(times, sz_expect_2, label='ref')
axs[0, 0].set_title('Spin Population')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('<Sz>')
axs[0, 0].legend()

# Oscillator occupation number plot
axs[0, 1].plot(times, occupation_numbers_1, label='new')
axs[0, 1].plot(times, occupation_numbers_2, label='ref')
axs[0, 1].set_title('Oscillator Occupation Number')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Occupation Number')
axs[0, 1].legend()

# Entanglement (Von Neumann Entropy) plot
axs[1, 0].plot(times, entanglement_1, label='new')
axs[1, 0].plot(times, entanglement_2, label='ref')
axs[1, 0].set_title('Entanglement')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Von Neumann Entropy')
axs[1, 0].legend()

# Leave the last plot empty
axs[1, 1].axis('off')

plt.tight_layout()
plt.savefig('explore_transformations.png')
