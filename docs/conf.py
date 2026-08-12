"""Sphinx configuration for the fishbonett documentation."""
from importlib.metadata import version as _version

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
