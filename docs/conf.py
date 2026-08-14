"""Sphinx configuration for the fishbonett documentation."""
import sys
from importlib.metadata import version as _version
from pathlib import Path

# Figures are build artefacts, not repository content: regenerate them (into the
# gitignored docs/img/) before the build so any checkout -- local, CI or RTD --
# renders against freshly computed data.  See docs/figures.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figures as _figures  # noqa: E402

_figures.build_all()

# The model/frame/propagator table is generated the same way and for the same
# reason: it is registry data, and a copy pasted into a page goes stale.  (One
# already has -- the hand-written table in docs/models/index.md predates two
# combinations the registry gained.)  getting_started.md literal-includes this.
from fishbonett.models.registry import describe_taxonomy  # noqa: E402

_generated = Path(__file__).resolve().parent / "_generated"
_generated.mkdir(exist_ok=True)
(_generated / "taxonomy.txt").write_text(describe_taxonomy() + "\n",
                                         encoding="utf-8")

project = "fishbonett"
copyright = "2020-2026, The fishbonett developers"
author = "The fishbonett developers"
try:
    release = _version("fishbonett")
except Exception:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False}
autodoc_mock_imports = ["cupy", "cupyx", "vegas"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["dollarmath", "amsmath"]

html_theme = "furo"
html_title = f"fishbonett {release}"
exclude_patterns = ["_build"]
