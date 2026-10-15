from pathlib import Path

from jingwei_benchmark.cli import parser
from jingwei_benchmark.runtime import default_input_path


def test_default_dataset_is_resolved_from_resources_not_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    default = default_input_path()
    assert default and default.name == "Alzheimer.csv"
    assert default.is_file()
    assert parser().parse_args(["inspect"]).input == default
    chosen = tmp_path / "chosen.csv"
    assert parser().parse_args(["inspect", "--input", str(chosen)]).input == chosen


def test_default_dataset_in_relocated_bundle(tmp_path, monkeypatch):
    import sys
    target = tmp_path / "data/Alzheimer.csv"
    target.parent.mkdir()
    target.write_text("A,B\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert default_input_path() == target
