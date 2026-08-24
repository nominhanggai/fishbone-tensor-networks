"""Cache behavior for generated documentation figures."""

import importlib.util
import os
from pathlib import Path
import sys
import time


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
