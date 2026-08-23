"""Small, repeatable algorithmic baseline for architecture changes.

Wall time is diagnostic.  The enforced metrics are numerical output, peak bond,
and Krylov work, which are stable across machines and catch accidental changes in
both accuracy and asymptotic work.
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fishbonett import Bath, SystemBath
from fishbonett.evolve._tdvp_kernels import _KRY
from fishbonett.operators import sigma_x, sigma_z

REFERENCE = Path(__file__).with_name("baseline_reference.json")


def measure():
    density = lambda w: 0.2 * w * np.exp(-w / 5.0)
    model = SystemBath(
        h=0.5 * sigma_x, coupling=sigma_z,
        bath=Bath(
            J=density, domain=(0.0, 30.0), n_modes=4, phys_dim=5))
    _KRY.update(calls=0, iters=0)
    start = perf_counter()
    result = model.run(
        dt=0.04, n_steps=6, method="interaction-star-tdvp2",
        trunc_eps=1e-8, bond_dim=16, observables={"sz": sigma_z}, seed=0)
    elapsed = perf_counter() - start
    return {
        "final_sz": float(result.expect["sz"][-1].real),
        "peak_bond": int(np.max(result.max_bond)),
        "krylov_calls": int(_KRY["calls"]),
        "krylov_iterations": int(_KRY["iters"]),
        "seconds": elapsed,
    }


def main():
    got = measure()
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert np.isclose(got["final_sz"], expected["final_sz"],
                      rtol=1e-10, atol=1e-12)
    for key in ("peak_bond", "krylov_calls", "krylov_iterations"):
        assert got[key] == expected[key], (key, got[key], expected[key])
    print(json.dumps(got, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
