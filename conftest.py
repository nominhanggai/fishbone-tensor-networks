"""Pytest bootstrap for running the suite from a source checkout.

Once the package is installed (``pip install -e .``) importing ``fishbonett``
works anywhere; this shim additionally makes the source tree and the
characterization helpers importable when running the tests in-place, before an
editable install has happened.
"""
import os
import sys

_ROOT = os.path.dirname(__file__)
for _p in (_ROOT, os.path.join(_ROOT, "tests", "characterization")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The legacy tests/*.py are exploratory scripts that execute (and plot) on import;
# they are not pytest tests and are excluded from collection until curated.
collect_ignore = [
    os.path.join("tests", "explore_transformations.py"),
    os.path.join("tests", "hsb_interaction_picture_single_mode.py"),
    os.path.join("tests", "hsb_interaction_picture_single_mode_qutip.py"),
]
collect_ignore_glob = [os.path.join("examples", "*")]
