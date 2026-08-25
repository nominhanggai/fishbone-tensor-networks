"""Cache behavior for generated documentation figures."""

import importlib.util
import os
from pathlib import Path
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_figures():
    spec = importlib.util.spec_from_file_location(
        "test_docs_figures_module", ROOT / "docs" / "figures.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stale_tutorial_figure_is_regenerated(tmp_path, monkeypatch):
    figures = _load_figures()
    name = "cache_probe"
    output = tmp_path / "img" / f"{name}.svg"
    summary = tmp_path / "generated" / f"{name}.md"
    output.parent.mkdir()
    summary.parent.mkdir()
    output.write_text("old figure", encoding="utf-8")
    summary.write_text("old summary", encoding="utf-8")

    # Leave enough margin for filesystems whose write timestamps have coarser
    # resolution than ``time.time_ns()``.
    source_time = time.time_ns() - 100_000_000
    old_time = source_time - 1_000_000_000
    os.utime(output, ns=(old_time, old_time))
    os.utime(summary, ns=(old_time, old_time))
    calls = []

    def generate(path):
        calls.append(path)
        path.write_text("new figure", encoding="utf-8")
        figures._write_summary(name, "new summary")
        return path

    monkeypatch.setattr(figures, "IMG", output.parent)
    monkeypatch.setattr(figures, "GENERATED", summary.parent)
    monkeypatch.setattr(figures, "_FIGURE_BY_NAME", {name: generate})
    monkeypatch.setattr(figures, "OUTPUTS", {name: output})
    monkeypatch.setattr(figures, "EXTRA_OUTPUTS", {})
    monkeypatch.setattr(figures, "_TUTORIAL_FIGURES", {name})
    monkeypatch.setattr(figures, "_input_mtime", lambda unused: source_time)

    assert figures.build_selected([name]) == [output]
    assert output.read_text(encoding="utf-8") == "new figure"
    assert summary.read_text(encoding="utf-8") == "new summary"
    assert figures.build_selected([name]) == []
    assert calls == [output]


def test_reference_data_are_figure_inputs():
    figures = _load_figures()
    reference = (
        ROOT / "examples" / "reference_data"
        / "dijkstra_2015_fig5_quantum_dynamics.csv"
    )
    assert figures._input_mtime("vibronic_dimer") >= reference.stat().st_mtime_ns


def test_unrelated_reference_data_do_not_stale_a_figure(tmp_path, monkeypatch):
    figures = _load_figures()
    reference_dir = tmp_path / "examples" / "reference_data"
    source_dir = tmp_path / "src" / "fishbonett"
    reference_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    target = reference_dir / "dijkstra_2015_fig5_quantum_dynamics.csv"
    unrelated = reference_dir / "nuomin_2022_fig8_ic10.csv"
    target.write_text("target", encoding="utf-8")
    unrelated.write_text("unrelated", encoding="utf-8")
    (tmp_path / "examples" / "vibronic_dimer.py").write_text(
        "", encoding="utf-8",
    )
    (source_dir / "module.py").write_text("", encoding="utf-8")

    target_time = time.time_ns() + 1_000_000_000
    unrelated_time = target_time + 1_000_000_000
    os.utime(target, ns=(target_time, target_time))
    os.utime(unrelated, ns=(unrelated_time, unrelated_time))
    monkeypatch.setattr(figures, "ROOT", tmp_path)

    assert figures._input_mtime("vibronic_dimer") == target_time


def test_layout_validation_rejects_overlapping_text():
    plt = pytest.importorskip("matplotlib.pyplot")
    figures = _load_figures()
    figure = plt.figure(figsize=(3, 2))
    figure.text(0.5, 0.5, "first label")
    figure.text(0.5, 0.5, "second label")

    with pytest.raises(RuntimeError, match="overlapping"):
        figures._validate_layout(figure, "overlap.svg")
    plt.close(figure)


def test_layout_validation_accepts_separated_text():
    plt = pytest.importorskip("matplotlib.pyplot")
    figures = _load_figures()
    figure = plt.figure(figsize=(3, 2))
    figure.text(0.1, 0.2, "left label")
    figure.text(0.7, 0.8, "right label")

    figures._validate_layout(figure, "clean.svg")
    plt.close(figure)
