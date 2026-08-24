"""The docs have to keep up with the code.

Documentation should be checked after every code change; this is the part of
that which a machine can check.  It exists because it was needed: three modules
added during the model/representation restructure (``fishbonett.system`` and
``fishbonett.representations.schrodinger``) were absent from the
API reference until a scan found them, and ``fishbonett.models.propagate`` went
missing the same way afterwards.
"""
import ast
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "fishbonett"
DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"

#: Modules the reference deliberately omits, and why.
NOT_IN_REFERENCE = {
    "fishbonett.rsvd_cupy": "optional GPU kernel behind the CuPy import guard",
}


def _modules():
    """Public *leaf* modules.

    Subpackage ``__init__`` files are excluded: a namespace is documented by its
    contents, and ``api.md`` lists the members rather than the package.  Their
    docstrings carry the layering prose, which the section intros in ``api.md``
    paraphrase.
    """
    out = set()
    for p in SRC.rglob("*.py"):
        if "__pycache__" in str(p) or p.name == "__init__.py":
            continue
        rel = p.relative_to(SRC.parent).with_suffix("").as_posix().replace("/", ".")
        if any(part.startswith("_") for part in rel.split(".")[1:]):
            continue
        out.add(rel)
    return out


def _listed():
    api = (DOCS / "api.md").read_text(encoding="utf-8")
    return set(re.findall(r"^\s+(fishbonett[\w.]*)\s*$", api, re.M))


def test_every_public_module_is_in_the_api_reference():
    """Listed outright, or covered by an ancestor package's ``:recursive:`` entry."""
    listed = _listed()

    def covered(mod):
        parts = mod.split(".")
        return any(".".join(parts[:k]) in listed for k in range(1, len(parts) + 1))

    missing = sorted(m for m in _modules()
                     if m != "fishbonett" and m not in NOT_IN_REFERENCE
                     and not covered(m))
    assert not missing, (
        "these modules are not reachable from docs/api.md: " + ", ".join(missing)
        + "\nAdd them to an autosummary block, or to NOT_IN_REFERENCE with a reason.")


def test_the_omission_list_is_accurate():
    """Every deliberately-omitted module must still exist, and stay omitted.

    Otherwise the list silently becomes a place where a real gap can hide.
    """
    mods = _modules()
    listed = _listed()
    for mod, reason in NOT_IN_REFERENCE.items():
        assert mod in mods, f"{mod} no longer exists; drop it from NOT_IN_REFERENCE"
        assert reason.strip(), f"{mod} needs a reason"
        assert mod not in listed, (
            f"{mod} is in the API reference now; remove it from NOT_IN_REFERENCE")


def test_api_reference_names_only_real_modules():
    """The reverse direction: no autosummary entry may point at nothing."""
    import importlib

    for name in sorted(_listed()):
        try:
            importlib.import_module(name)
        except ImportError as exc:  # pragma: no cover - only on a real break
            pytest.fail(f"docs/api.md lists {name}, which does not import: {exc}")


# -- the metadata has to agree with itself -----------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _min_python():
    """The oldest interpreter ``pyproject.toml`` claims to support."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+)\.(\d+)', text)
    assert m, "pyproject.toml has no parseable requires-python"
    return int(m.group(1)), int(m.group(2))


def test_sources_parse_against_the_oldest_supported_python():
    """Nothing may use syntax newer than ``requires-python`` allows.

    CI catches this, but only after a full matrix run, and only for whoever pushes.
    A developer on a recent interpreter can write newer syntax and see nothing wrong
    locally.  ``ast`` can check it directly, in a fraction of a second.
    """
    floor = _min_python()
    # the check must be capable of failing, or it proves nothing
    with pytest.raises(SyntaxError):
        ast.parse("type X = int\n", feature_version=(3, 10))

    bad = []
    for root in ("src/fishbonett", "tests", "examples", "benchmarks"):
        for p in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in str(p) or "legacy" in p.parts:
                continue
            try:
                ast.parse(p.read_text(encoding="utf-8"), feature_version=floor)
            except SyntaxError as exc:
                rel = p.relative_to(ROOT).as_posix()
                bad.append(f"{rel}:{exc.lineno} {exc.msg}")
    assert not bad, (
        f"syntax newer than Python {floor[0]}.{floor[1]}, which pyproject claims to "
        "support:\n  " + "\n  ".join(bad))


def test_declared_python_support_is_consistent():
    """``requires-python``, the classifiers and the CI matrix must agree.

    Three places say which interpreters are supported; nothing was checking that
    they say the same thing.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    ci_path = ROOT / ".github" / "workflows" / "ci.yml"
    if not ci_path.exists():
        pytest.skip("CI workflow is intentionally absent from source distributions")
    ci = ci_path.read_text(encoding="utf-8")

    classifiers = {v for v in re.findall(
        r'"Programming Language :: Python :: ([\d.]+)"', text) if "." in v}
    matrix = set(re.findall(r'"(\d+\.\d+)"',
                            re.search(r"python-version:\s*\[([^\]]+)\]", ci).group(1)))
    assert classifiers == matrix, (
        f"pyproject classifiers {sorted(classifiers)} != CI matrix {sorted(matrix)}")

    floor = _min_python()
    assert f"{floor[0]}.{floor[1]}" in matrix, (
        f"requires-python floor {floor} is not in the CI matrix {sorted(matrix)}")


def _prose_files():
    out = [p for p in DOCS.rglob("*.md")] + [ROOT / "README.md"]
    return [p for p in out if "generated" not in p.parts]


def _source_docstrings():
    """Yield source docstrings with their file and line number."""
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef,
                                     ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            value = ast.get_docstring(node, clean=False)
            if value is not None:
                yield path, getattr(node, "lineno", 1), value


def test_docs_do_not_contain_refactor_commentary():
    """Published prose describes the current API, not the refactor that produced it."""
    banned = (
        "there is no second public category",
        "single source of truth",
        "used to live",
        "used to point into",
        "used to share no code",
        "what used to be",
        "what this buys",
        "the ordering is the point",
        "standing confusion",
        "the fix is",
        "near-identical",
        "were unreachable",
        "replaced seven",
    )
    sources = [
        (path, 1, path.read_text(encoding="utf-8", errors="ignore"))
        for path in _prose_files()
    ]
    sources.extend(_source_docstrings())
    found = []
    for path, start, prose in sources:
        lowered = prose.lower()
        for phrase in banned:
            if phrase in lowered:
                line = start + lowered[:lowered.index(phrase)].count("\n")
                found.append(f"{path.relative_to(ROOT)}:{line}: {phrase}")
    assert not found, "refactor commentary in published documentation:\n  " + "\n  ".join(found)


def test_doc_code_blocks_import_things_that_exist():
    """Every ``from fishbonett... import X`` in a doc snippet must resolve.

    Doc examples rot silently when things are renamed, and a lot was renamed in the
    model/representation restructure.  This caught ``docs/bath.md`` importing ``thermalize``
    from ``fishbonett.models``, where it has never lived -- anyone copying that
    snippet got an ImportError.
    """
    import importlib

    bad = []
    for p in _prose_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"```python\n(.*?)```", text, re.S):
            line = text[: m.start()].count("\n") + 1
            try:
                tree = ast.parse(m.group(1))
            except SyntaxError as exc:
                bad.append(f"{p.name}:{line} does not parse: {exc.msg}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("fishbonett"):
                    try:
                        mod = importlib.import_module(node.module)
                    except ImportError as exc:
                        bad.append(f"{p.name}:{line} cannot import {node.module}: {exc}")
                        continue
                    for alias in node.names:
                        if not hasattr(mod, alias.name):
                            bad.append(f"{p.name}:{line} {node.module} has no {alias.name!r}")
    assert not bad, "broken imports in documentation examples:\n  " + "\n  ".join(bad)


def test_cross_references_point_at_real_objects():
    """``:func:`fishbonett.x.y``` and friends must name something that exists.

    Sphinx only reports these under ``-n``, which the build does not use because
    numpydoc type words ("optional", "array", "callable") would drown the signal.
    This checks just our own targets, where an unresolved name is always a bug.
    """
    import importlib

    role = re.compile(r":(?:py:)?(mod|func|class|meth|attr|data|obj):`~?([^`<]*?)`")

    def resolves(target):
        target = target.strip().rstrip("()")
        try:
            importlib.import_module(target)
            return True
        except ImportError:
            pass
        if "." not in target:
            return False
        mod, _, attr = target.rpartition(".")
        while mod:
            try:
                obj = importlib.import_module(mod)
            except ImportError:
                mod, _, head = mod.rpartition(".")
                attr = f"{head}.{attr}"
                continue
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    return False
            return True
        return False

    sources = [p for p in SRC.rglob("*.py") if "__pycache__" not in str(p)]
    bad = []
    for p in sources + _prose_files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in role.finditer(text):
            target = m.group(2)
            if target.startswith("fishbonett") and not resolves(target):
                line = text[: m.start()].count("\n") + 1
                bad.append(f"{p.name}:{line} :{m.group(1)}:`{target}`")
    assert not bad, "cross-references naming nothing:\n  " + "\n  ".join(bad)
