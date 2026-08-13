"""Numerical regression ("characterization") tests.

Golden observable arrays were captured from the pre-refactor code (see
``scenario.py``).  These tests re-run the identical scenario against the current
code and assert the physical observables are unchanged, which is what guards the
"unify + preserve" refactor of the TEBD engine and helpers.

The ``_run`` import lines are the single place that tracks where the model/engine
classes live; they are updated as modules are relocated during the refactor,
while the golden *values* stay fixed.
"""
import os

import numpy as np
import pytest

from scenario import run_multichannel_ic

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "_golden")
_KEYS = ("spin_rho", "pop_z", "boson_num_final")


def _load(name):
    return {k: np.load(os.path.join(GOLDEN_DIR, f"mc_{name}_{k}.npy")) for k in _KEYS}


def _run(name):
    from fishbonett.operators import sigma_x, sigma_z, number
    from fishbonett.states.mps import SystemBathMPS
    from fishbonett.frames.multichannel import SystemBathMultiChannel as SystemBath
    return run_multichannel_ic(SystemBath, SystemBathMPS, sigma_x, sigma_z, number,
                               lbo=(name == "lbo"))


@pytest.mark.parametrize("name", ["plain", "lbo"])
def test_multichannel_ic_matches_golden(name):
    got = _run(name)
    exp = _load(name)
    for k in _KEYS:
        np.testing.assert_allclose(got[k], exp[k], rtol=1e-6, atol=1e-8,
                                   err_msg=f"{name}/{k} drifted from golden")
