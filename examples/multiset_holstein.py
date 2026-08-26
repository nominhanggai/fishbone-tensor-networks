"""Franck--Condon dynamics in a finite Holstein chain.

The same Hamiltonian is propagated with the multi-set MPS ansatz of Kloss,
Reichman, and Tempelaar and with a conventional impurity-plus-modes MPS.  This
is a method comparison, not a digitization of the paper's plotted curves.

Run a quick calculation with::

    python examples/multiset_holstein.py --profile quick

The output NPZ stores populations, root-mean-square displacements, and retained
bonds for both state ansaetze.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fishbonett.evolve.multiset import run_multiset_mpo_hamiltonian
from fishbonett.evolve.tdvp import measure_rdm, run_mpo_hamiltonian
from fishbonett.operators import annihilate
from fishbonett.representations._mpo import identity_product, product_sum_mpo
from fishbonett.states.multiset import MultiSetMPS


@dataclass(frozen=True)
class Profile:
    sites: int
    fock: int
    coupling: float
    dt: float
    t_max: float
    trunc_eps: float


PROFILES = {
    "smoke": Profile(3, 4, 1.0, 0.04, 0.08, 1e-10),
    "quick": Profile(5, 8, 1.5, 0.04, 0.8, 1e-6),
    "paper-scale": Profile(31, 32, 2.5, 0.025, 6 * np.pi, 1e-6),
}


class HolsteinRepresentation:
    """Static full-MPO representation of a one-particle Holstein chain."""

    static = True

    def __init__(self, sites, fock, vibronic_coupling, *, hopping=1.0, frequency=1.0):
        self.sites = int(sites)
        self.fock = int(fock)
        self.g = float(vibronic_coupling)
        self.hopping = float(hopping)
        self.frequency = float(frequency)
        self.dimensions = (self.sites,) + (self.fock,) * self.sites
        self._mpo = self._build_mpo()

    def _build_mpo(self):
        products = []
        coefficients = []
        destroy = annihilate(self.fock)
        number = destroy.conj().T @ destroy
        displacement = destroy + destroy.conj().T
        for mode in range(self.sites):
            row = identity_product(self.dimensions)
            row[mode + 1] = number
            products.append(row)
            coefficients.append(self.frequency)

            projector = np.zeros((self.sites, self.sites), complex)
            projector[mode, mode] = 1.0
            row = identity_product(self.dimensions)
            row[0] = projector
            row[mode + 1] = displacement
            products.append(row)
            coefficients.append(self.g * self.frequency)
        for left in range(self.sites - 1):
            hopping = np.zeros((self.sites, self.sites), complex)
            hopping[left, left + 1] = self.hopping
            hopping[left + 1, left] = self.hopping
            row = identity_product(self.dimensions)
            row[0] = hopping
            products.append(row)
            coefficients.append(1.0)
        return product_sum_mpo(self.dimensions, products, coefficients)

    def tdvp_mpo(self, _time=None):
        return self._mpo


def _rms_displacement(populations):
    centre = (populations.shape[1] - 1) // 2
    distance_squared = (np.arange(populations.shape[1]) - centre) ** 2
    return np.sqrt(np.maximum(0.0, populations @ distance_squared))


def run(profile):
    representation = HolsteinRepresentation(profile.sites, profile.fock, profile.coupling)
    initial = np.zeros(profile.sites, complex)
    initial[profile.sites // 2] = 1.0
    steps = int(round(profile.t_max / profile.dt))
    common = dict(
        dt=profile.dt,
        nsteps=steps,
        bond_dim=None,
        trunc_eps=profile.trunc_eps,
        krylov=30,
        tol=1e-8,
    )

    started = perf_counter()
    multi = run_multiset_mpo_hamiltonian(
        representation,
        state=MultiSetMPS.product(initial, representation.dimensions[1:]),
        **common,
    )
    multi_seconds = perf_counter() - started
    multi_t, multi_rdm, multi_bond, set_bonds, _state = multi

    started = perf_counter()
    conventional_t, conventional_rdm, conventional_bond = run_mpo_hamiltonian(
        representation,
        sweep="tdvp2",
        initial=initial,
        observe=lambda tensors: measure_rdm(tensors[0]),
        **common,
    )
    conventional_seconds = perf_counter() - started

    multi_population = np.real(np.diagonal(multi_rdm, axis1=1, axis2=2))
    conventional_population = np.real(np.diagonal(conventional_rdm, axis1=1, axis2=2))
    arrays = {
        "t": multi_t,
        "multiset_population": multi_population,
        "multiset_rms": _rms_displacement(multi_population),
        "multiset_max_bond": multi_bond,
        "multiset_set_bonds": set_bonds,
        "conventional_population": conventional_population,
        "conventional_rms": _rms_displacement(conventional_population),
        "conventional_max_bond": conventional_bond,
    }
    summary = {
        "sites": profile.sites,
        "fock": profile.fock,
        "g": profile.coupling,
        "dt": profile.dt,
        "t_max": profile.t_max,
        "trunc_eps": profile.trunc_eps,
        "multiset_seconds": multi_seconds,
        "conventional_seconds": conventional_seconds,
        "multiset_peak_bond": int(np.max(multi_bond)),
        "conventional_peak_bond": int(np.max(conventional_bond)),
        "maximum_population_difference": float(
            np.max(np.abs(multi_population - conventional_population))
        ),
    }
    if not np.allclose(multi_t, conventional_t):
        raise RuntimeError("the compared calculations used different time grids")
    return arrays, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multiset_holstein.npz"),
    )
    args = parser.parse_args()
    arrays, summary = run(PROFILES[args.profile])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
