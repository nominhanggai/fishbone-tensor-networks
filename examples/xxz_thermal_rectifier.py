"""A four-spin XXZ molecular junction between two explicit thermal baths.

The quick command used by the examples smoke test is::

    python examples/xxz_thermal_rectifier.py

The documentation build calls ``run_suite(profile="full")``.  That calculation
uses automatically resolved T-TEDOPA baths and performs forward/reverse bias,
an equilibrium control, and numerical convergence checks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from fishbonett import (
    Bath, Fishbone, GibbsPurification, energy_current_operator,
    sigma_x, sigma_y, sigma_z,
)


@dataclass(frozen=True)
class RectifierConfig:
    delta: float = 4.0
    interface: float = 0.5
    field: float = 0.5
    bath_strength: float = 0.2
    cutoff: float = 2.0
    cold: float = 0.5
    hot: float = 5.0
    wire_temperature: float = 0.5
    phys_dim: int = 6
    dt: float = 0.02
    horizon: float = 4.0
    bath_horizon: float = 4.0
    trunc_eps: float = 1e-5
    max_bond: int = 512


def _physical_terms(config):
    sx, sy, sz = sigma_x / 2, sigma_y / 2, sigma_z / 2
    sites = [config.field * sz for _ in range(4)]
    left = np.kron(sx, sx) + np.kron(sy, sy) + config.delta * np.kron(sz, sz)
    interface = config.interface * (np.kron(sx, sx) + np.kron(sy, sy))
    right = np.kron(sx, sx) + np.kron(sy, sy)
    return sites, [left, interface, right]


def make_rectifier(config, temperatures):
    """Return ``(model, purification, observables)`` for one bath bias."""
    sites, backbone = _physical_terms(config)
    purification = GibbsPurification(
        sites, backbone, temperature=config.wire_temperature)

    def spectral_density(frequency):
        return (config.bath_strength * frequency
                * np.exp(-(frequency / config.cutoff) ** 2))

    baths = [Bath(
        J=spectral_density, temperature=temperature,
        phys_dim=config.phys_dim, discretization="tedopa")
        for temperature in temperatures]
    coupling_left = purification.lift_operator(sigma_x, [0])
    coupling_right = purification.lift_operator(sigma_x, [3])
    model = Fishbone(
        sites=purification.sites,
        backbone=purification.backbone,
        baths=[baths[0].bind(coupling_left), None, None,
               baths[1].bind(coupling_right)],
    )

    currents = [
        energy_current_operator(sites[0], backbone[0]),
        energy_current_operator(sites[1], backbone[1], backbone[0]),
        energy_current_operator(sites[2], backbone[2], backbone[1]),
    ]
    observables = {
        "current_0": (purification.lift_operator(currents[0], [0, 1]), (0, 1)),
        "current_1": (
            purification.lift_operator(currents[1], [0, 1, 2]), (0, 1, 2)),
        "current_2": (
            purification.lift_operator(currents[2], [1, 2, 3]), (1, 2, 3)),
        "magnetization": purification.lift_operator(sigma_z, [0]),
    }
    for site, hamiltonian in enumerate(sites):
        observables[f"site_energy_{site}"] = (
            purification.lift_operator(hamiltonian, [site]), site)
    return model, purification, observables


def _join(first, second):
    return {
        "t": np.concatenate([first.t, second.t]),
        "expect": {
            name: np.concatenate([first.expect[name], second.expect[name]])
            for name in first.expect
        },
        "max_bond": np.concatenate([first.max_bond, second.max_bond]),
        "checkpoint": second.checkpoint,
        "method": second.method,
    }


def run_bias(config, temperatures, *, demonstrate_restart=True):
    """Run one bias, optionally as two checkpoint-continuation segments."""
    model, purification, observables = make_rectifier(config, temperatures)
    n_steps = int(round(config.horizon / config.dt))
    common = dict(
        dt=config.dt, trunc_eps=config.trunc_eps,
        bond_dim=config.max_bond, observables=observables,
        bath_horizon=config.bath_horizon,
        observe_every=max(1, int(round(0.1 / config.dt))),
    )
    if not demonstrate_restart:
        result = model.run(n_steps=n_steps, initial=purification, **common)
        return {"t": result.t, "expect": result.expect,
                "max_bond": result.max_bond, "checkpoint": result.checkpoint,
                "method": result.method}
    split = n_steps // 2
    first = model.run(n_steps=split, initial=purification, **common)
    second = model.run(n_steps=n_steps - split, resume=first.checkpoint, **common)
    return _join(first, second)


def _current_summary(trajectory, fraction=0.25):
    time = trajectory["t"]
    select = time >= time[-1] * (1.0 - fraction)
    means, drifts = [], []
    for cut in range(3):
        values = trajectory["expect"][f"current_{cut}"][select]
        means.append(float(np.mean(values)))
        if len(values) > 1:
            slope = np.polyfit(time[select], values, 1)[0]
            scale = max(abs(np.mean(values)), 1e-12)
            drifts.append(float(abs(slope) * np.ptp(time[select]) / scale))
        else:
            drifts.append(np.inf)
    return np.asarray(means), np.asarray(drifts)


def run_suite(profile="quick"):
    """Run the demonstration or the complete documentation validation suite."""
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'")
    if profile == "quick":
        config = RectifierConfig(
            phys_dim=3, dt=0.04, horizon=0.4, bath_horizon=0.4,
            max_bond=128)
        forward = run_bias(config, (config.hot, config.cold),
                           demonstrate_restart=False)
        return {"config": config, "forward": forward}

    base = RectifierConfig()
    suite = {
        "config": base,
        "forward": run_bias(base, (base.hot, base.cold)),
        "reverse": run_bias(base, (base.cold, base.hot)),
        "equilibrium": run_bias(base, (base.cold, base.cold)),
        "convergence": {},
    }
    # Converge the resolved transient on a common shorter interval. If the main
    # runs ever pass the stationarity gates, final-window convergence must be
    # added before a rectification ratio can be published.
    convergence_base = RectifierConfig(
        **{**base.__dict__, "horizon": 2.0, "bath_horizon": 2.0})
    suite["convergence"]["reference"] = {
        "forward": run_bias(
            convergence_base, (convergence_base.hot, convergence_base.cold)),
        "reverse": run_bias(
            convergence_base, (convergence_base.cold, convergence_base.hot)),
    }
    variants = {
        "half_dt": RectifierConfig(
            **{**convergence_base.__dict__, "dt": convergence_base.dt / 2}),
        "tight_svd": RectifierConfig(
            **{**convergence_base.__dict__, "trunc_eps": 1e-6}),
        "larger_fock": RectifierConfig(
            **{**convergence_base.__dict__, "phys_dim": 7}),
        "larger_bath": RectifierConfig(
            **{**convergence_base.__dict__, "bath_horizon": 4.0}),
    }
    for name, config in variants.items():
        suite["convergence"][name] = {
            "forward": run_bias(config, (config.hot, config.cold)),
            "reverse": run_bias(config, (config.cold, config.hot)),
        }
    return suite


def summarize(suite):
    """Numerical summary used by both the terminal example and the docs."""
    forward, forward_drift = _current_summary(suite["forward"])
    if "reverse" not in suite:
        return {"forward_currents": forward,
                "forward_drift": forward_drift,
                "max_bond": int(np.max(suite["forward"]["max_bond"]))}
    reverse, reverse_drift = _current_summary(suite["reverse"])
    equilibrium, equilibrium_drift = _current_summary(suite["equilibrium"])
    jf, jr = float(np.mean(forward)), float(np.mean(reverse))
    cut_scale = max(abs(jf), abs(jr), 1e-15)
    cut_agreement = {
        "forward": float(np.ptp(forward) / cut_scale),
        "reverse": float(np.ptp(reverse) / cut_scale),
    }
    stationary = (max(cut_agreement.values()) <= 0.1
                  and max(forward_drift.max(), reverse_drift.max()) <= 0.1)
    equilibrium_small = float(np.max(np.abs(equilibrium))) <= 0.1 * min(
        max(abs(jf), 1e-15), max(abs(jr), 1e-15))
    convergence = {}
    convergence_runs = suite.get("convergence", {})
    if convergence_runs:
        reference_forward = float(
            _current_summary(convergence_runs["reference"]["forward"])[0][1])
        reference_reverse = float(
            _current_summary(convergence_runs["reference"]["reverse"])[0][1])
    for name, values in convergence_runs.items():
        if name == "reference":
            continue
        vf = float(_current_summary(values["forward"])[0][1])
        vr = float(_current_summary(values["reverse"])[0][1])
        convergence[name] = {
            "forward_relative": abs(vf - reference_forward)
                                / max(abs(reference_forward), 1e-15),
            "reverse_relative": abs(vr - reference_reverse)
                                / max(abs(reference_reverse), 1e-15),
        }
    numerical_convergence = all(
        max(values.values()) <= 0.05 for values in convergence.values())
    ratio = (abs(jf) / max(abs(jr), 1e-15)
             if stationary and equilibrium_small and numerical_convergence
             else None)
    return {
        "forward_currents": forward, "reverse_currents": reverse,
        "equilibrium_currents": equilibrium,
        "forward_drift": forward_drift, "reverse_drift": reverse_drift,
        "equilibrium_drift": equilibrium_drift,
        "forward": jf, "reverse": jr, "rectification": ratio,
        "stationary": stationary, "equilibrium_small": equilibrium_small,
        "numerical_convergence": numerical_convergence,
        "cut_agreement": cut_agreement, "convergence": convergence,
        "max_bond": max(
            int(np.max(suite[key]["max_bond"]))
            for key in ("forward", "reverse", "equilibrium")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    summary = summarize(run_suite(args.profile))
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
