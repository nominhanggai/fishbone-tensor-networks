"""Small, repeatable algorithmic baseline for architecture changes.

The propagated observable and Krylov call count are strict regression checks.
Peak bond and Krylov iteration count have narrow tolerances because floating-point
contraction order and BLAS implementations can change threshold decisions.  Wall
time is diagnostic only.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fishbonett import Bath, SystemBath
from fishbonett.evolve.tdvp import krylov_statistics
from fishbonett.operators import sigma_x, sigma_z

REFERENCE = Path(__file__).with_name("baseline_reference.json")


def measure():
    density = lambda w: 0.2 * w * np.exp(-w / 5.0)
    model = SystemBath(
        h=0.5 * sigma_x, coupling=sigma_z,
        bath=Bath(
            J=density, domain=(0.0, 30.0), n_modes=4, phys_dim=5))
    krylov_statistics(reset=True)
    start = perf_counter()
    result = model.run(
        dt=0.04, n_steps=6, method="interaction-chain-tdvp2",
        trunc_eps=1e-8, bond_dim=16, observables={"sz": sigma_z}, seed=0)
    elapsed = perf_counter() - start
    return {
        "final_sz": float(result.expect["sz"][-1].real),
        "peak_bond": int(np.max(result.max_bond)),
        "krylov_calls": int(krylov_statistics()["calls"]),
        "krylov_iterations": int(krylov_statistics()["iters"]),
        "seconds": elapsed,
    }


def main():
    got = measure()
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert np.isclose(got["final_sz"], expected["final_sz"],
                      rtol=1e-10, atol=1e-12)
    assert got["krylov_calls"] == expected["krylov_calls"], (
        "krylov_calls", got["krylov_calls"], expected["krylov_calls"])
    for key in ("peak_bond", "krylov_iterations"):
        tolerance = expected[f"{key}_tolerance"]
        assert abs(got[key] - expected[key]) <= tolerance, (
            key, got[key], expected[key], tolerance)
    print(json.dumps(got, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
