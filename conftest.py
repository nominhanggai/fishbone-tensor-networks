"""Make the source package and characterization helpers importable in tests."""
import os
import sys

_ROOT = os.path.dirname(__file__)
for _p in (
    os.path.join(_ROOT, "src"),
    os.path.join(_ROOT, "tests", "characterization"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
