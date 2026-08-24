"""Distribution metadata and numerical-method citations stay synchronized."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_distribution_declares_mit_license():
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = "MIT"' in metadata
    assert 'license-files = ["LICENSE"]' in metadata


# Scientific citations are part of the implementation documentation and must
# remain attached to the algorithms they define.
CITED = {
    "src/fishbonett/evolve/_tdvp_kernels.py": ("Haegeman", "Phys. Rev. B"),
    "src/fishbonett/bath/tedopa.py": ("Golub", "Lanczos"),
}


def test_numerical_engines_cite_the_published_algorithms():
    for relative, markers in CITED.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        assert not missing, f"{relative} lost its algorithm citation: {missing}"


def test_mit_license_is_complete():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Permission is hereby granted" in license_text
