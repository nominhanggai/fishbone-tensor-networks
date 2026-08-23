# Vibrationally assisted transfer in a molecular dimer

This tutorial asks a chemically useful question: can a damped intramolecular
vibration accelerate excitation transfer between two electronically mismatched
molecules? It reproduces the model studied by
[Dijkstra *et al.*](https://arxiv.org/abs/1309.4910), while using a tensor-network
bath rather than a reduced master equation.

You should be comfortable with open-system Hamiltonians and spectral densities;
no tensor-network implementation knowledge is assumed.

## Physical model

In the one-excitation basis $\{|D\rangle,|A\rangle\}$,

$$
H_S/J = \begin{pmatrix}8&-1\\-1&0\end{pmatrix}.
$$

Each molecule has an independent bath coupled to its local excitation projector.
The Brownian-oscillator density is

$$
J(\omega)=\frac{2\lambda\gamma\omega_0^2\omega}
{(\omega_0^2-\omega^2)^2+\gamma^2\omega^2},
\qquad \lambda=0.2J,\quad\gamma=2J/3,
$$

at $T=10J$ ($\beta J=0.1$). We scan $\omega_0$ because transfer can respond both
to the bath relaxation scale and to the $8J$ electronic energy gap.

```python
electronic = np.array([[8.0, -1.0], [-1.0, 0.0]])
baths = [make_bath(omega_0, profile).bind(occupied) for _ in range(2)]
model = Fishbone.from_single_excitation(electronic, baths=baths)
```

`from_single_excitation` maps the two-state electronic Hamiltonian to two local
two-level sites. The initial product state has the donor occupied and the
acceptor empty. A bare occupancy observable is measured on both sites.

## Why this numerical method?

The calculation uses `interaction-chain-fishbone-trotter-mpo`. The free star-bath
Hamiltonian is put in the interaction picture and the resulting time-dependent
coupling is transformed from star to chain coordinates. This is particularly
useful here because a strong, thermally occupied structured bath needs a generous
local Fock space, while the interaction representation often limits bath-induced
state entanglement.

Continuous parts are discretized with TEDOPA quadrature. Both scientific profiles
use the automatic reorganization-energy frequency domain. The CI-bounded
documentation profile uses 40 modes; the manual reference profile lets the
light-cone resolver choose the required count.

| profile | purpose | time step | horizon | local Fock dimension | frequencies |
|---|---|---:|---:|---:|---|
| `smoke` | API/engine check only | $0.01/J$ | $0.04/J$ | 3 | $8J$ |
| `docs` | plotted dynamics, 24 modes/bath | $0.05/J$ | $10/J$ | 8 | $4,8J$ |
| `reference` | manual convergence study | $0.001/J$ | $20/J$ | 20 | $2\ldots10J$ |

Every profile uses `bond_dim=None`; the SVD threshold, not a small bond cap, sets
the retained state space. Run the inexpensive profile with

```bash
python examples/vibronic_dimer.py
python examples/vibronic_dimer.py --profile docs \
  --output examples/output/vibronic_dimer_docs.npz
```

## Dynamics and conclusion

![Acceptor population for the vibrational-frequency scan](../img/vibronic_dimer.png)

```{include} ../_generated/vibronic_dimer.md
```

Two checks matter before interpreting the scan. First, donor plus acceptor
population must remain one. Second, the apparent enhancement must survive a
smaller time step, a tighter SVD threshold, and a larger local Fock space. The
reference profile is designed for those checks. A four-point documentation scan
can show the two physically motivated frequency regions, but it cannot locate a
sharp optimum or establish a universal resonance rule.
