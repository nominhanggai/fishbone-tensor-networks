import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

# Constants
v = 0.5  # Example value
omega = 1  # Example value
g = 2  # Example value
hbar = 1  # Planck's constant (set to 1 for simplicity)

# Parameters for simulation
N = 20  # Size of the Hilbert space for the oscillator
dt = 0.01  # Small time step
total_time = 40  # Total time for simulation
n_steps = int(total_time / dt)

# Operators
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


def a_operator(N):
    """Creation and annihilation operators for N-dimensional Hilbert space."""
    a = np.zeros((N, N), dtype=complex)
    for n in range(N-1):
        a[n, n + 1] = np.sqrt(n+1)
    return a

def u(h, t):
    return la.expm(-1j * h * t / hbar)

# def hamiltonian_1(t, N):
#     """First time-dependent Hamiltonian."""
#     a = a_operator(N)
#     a_dagger = a.T.conj()
#     exp_term = la.expm(-2j * t * g * np.kron(sigma_z, a_dagger+a))
#     ham = v * np.kron(sigma_x, np.eye(N)) @ exp_term  \
#           + omega * np.kron(np.eye(2), a_dagger@a) \
#           + 1j * omega * t * g * np.kron(sigma_z, a-a_dagger) - omega*(1j*t*g)**2 * np.kron(np.eye(2), np.eye(N))
#     # check if ham is hermitian
#     assert np.allclose(ham, ham.T.conj())
#     return ham

def hamiltonian_1(t, N):
    """First time-dependent Hamiltonian."""
    a = a_operator(N)
    a_dagger = a.T.conj()
    hs = v * np.kron(sigma_x, np.eye(N))
    hb = omega * np.kron(np.eye(2), np.dot(a_dagger, a))
    hsb = g * np.kron(sigma_z, (a_dagger + a))
    h = hs + hb + hsb
    # ham = h
    # ham = u(hsb, 1) @ h @ u(hsb, -1)
    # ham = la.expm(np.kron(sigma_z, a_dagger-a)) @ h @ la.expm(np.kron(sigma_z, a-a_dagger))
    trans_op = la.expm(g/omega * np.kron(sigma_z, 
                                         a * (np.exp(1j*omega*t)-1) - a.T * (np.exp(-1j*omega*t)-1)
                                         ))
    # trans_op = la.expm(-g/omega  * np.kron(sigma_z, a-a_dagger))
    # trans_op = la.expm(-g/omega  * np.kron(np.eye(2), a-a_dagger))
    # # trans_op = la.expm(-g/omega  * np.kron(sigma_z, 1j*np.eye(N)))
    # # check unitary
    # assert np.allclose(trans_op.T.conj()@trans_op, np.eye(2*N))
    # # check if ham is hermitian
    # ham =  trans_op.T.conj()@hs@trans_op + hb
    exp_term = la.expm(-2 * g/omega * np.kron(sigma_z, 
                                         a * (np.exp(1j*omega*t)-1) - a.T * (np.exp(-1j*omega*t)-1)
                                         ))
    ham = v * np.kron(sigma_x, np.eye(N)) @ exp_term + hb
    assert np.allclose(ham, ham.T.conj())
    return ham

def hamiltonian_2(N):
    """Second time-independent Hamiltonian."""
    a = a_operator(N)
    a_dagger = a.T.conj()
    ham = v * np.kron(sigma_x, np.eye(N)) \
          + omega * np.kron(np.eye(2), np.dot(a_dagger, a)) \
          + g * np.kron(sigma_z, (a_dagger + a))
    return ham


def evolve(state, t, dt, N, time_dependent=True):
    """Evolve the state by a small time step dt using the Hamiltonian."""
    H = hamiltonian_1(t, N) if time_dependent else hamiltonian_2(N)
    U = la.expm(-1j * H * dt / hbar)
    state_dt = U @ state
    return state_dt


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
for step in range(n_steps):
    print(f'Step {step+1}/{n_steps}')
    t = step * dt
    psi_t_1 = evolve(psi_t_1, t, dt, N, time_dependent=True)
    psi_t_2 = evolve(psi_t_2, t, dt, N, time_dependent=False)
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
plt.savefig('hsb_interaction_picture_single_mode.png')
