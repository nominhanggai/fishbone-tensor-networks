"""The high-level interface: spin-boson and fishbone dynamics in a few lines.

The low-level engines require the user to hand-write a TEBD/TDVP sweep loop (see
the other examples).  The :class:`~fishbonett.simulate.SystemBath` and
:class:`~fishbonett.simulate.Fishbone` classes wrap that away: declare the
bath(s) and system, then call ``run`` once.

Run with:  python examples/friendly_interface.py
"""
import numpy as np

from fishbonett import Bath, SystemBath, Fishbone
from fishbonett.operators import sigma_x, sigma_z


def spin_boson():
    """A single two-level system coupled to one bath, several engines."""
    bath = Bath(J=lambda w: 0.2 * w * np.exp(-w / 5.0),
                domain=(-25.0, 36.0), temperature=1.0,
                n_modes=6, phys_dim=6, discretization="orthpol")
    model = SystemBath(h=sigma_x, coupling=sigma_z, bath=bath)
    for method in ("tebd", "mpo-tdvp1", "mpo-tdvp2", "tree-tebd"):
        res = model.run(dt=0.05, t_max=0.5, method=method, bond_dim=30,
                        observables={"sz": sigma_z})
        print(f"  {method:11s} <sz>(t_end) = {res.expect['sz'][-1]:+.4f}")


def fishbone():
    """A 1D chain of 3 electronic sites, each with two baths (fishbone)."""
    J = lambda w: 0.2 * w * np.exp(-w / 5.0)

    def bath(op):
        return Bath(J=J, domain=(0.0, 40.0), n_modes=4, phys_dim=5, coupling=op)

    fb = Fishbone(
        sites=[0.5 * sigma_z + sigma_x] * 3,                 # 3 two-level sites
        baths=[(bath(sigma_z), bath(sigma_x))] * 3,          # two baths per site
        backbone=[0.4 * np.kron(sigma_z, sigma_z)] * 2,      # nearest-neighbour
    )
    # trunc_eps sets the accuracy: an interior two-bath site is a high-degree tree
    # tensor, so an over-tight eps inflates its bonds for negligible accuracy gain.
    res = fb.run(dt=0.02, t_max=0.2, bond_dim=30, trunc_eps=1e-7,
                 observables={"sz": sigma_z})
    print("  <sz>(t_end) per site:", np.round(res.expect["sz"][-1], 4))


if __name__ == "__main__":
    print("Spin-boson (one bath, several methods -- all agree):")
    spin_boson()
    print("Fishbone (three sites, two baths each):")
    fishbone()
