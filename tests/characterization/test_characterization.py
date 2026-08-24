"""Golden-value tests for a deterministic multichannel propagation."""
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
    from fishbonett.representations.multichannel import (
        MultichannelInteractionRepresentation,
    )
    from fishbonett.states.mps import SystemBathMPS

    return run_multichannel_ic(
        MultichannelInteractionRepresentation,
        SystemBathMPS,
        sigma_x,
        sigma_z,
        number,
        lbo=(name == "lbo"),
    )


@pytest.mark.parametrize("name", ["plain", "lbo"])
def test_multichannel_ic_matches_golden(name):
    got = _run(name)
    exp = _load(name)
    for k in _KEYS:
        np.testing.assert_allclose(got[k], exp[k], rtol=1e-6, atol=1e-8,
                                   err_msg=f"{name}/{k} drifted from golden")
