import json

from qft_benchmark import main, run_qft_benchmark


def test_qft_benchmark_writes_truthful_reference_artifacts(tmp_path):
    report = run_qft_benchmark(tmp_path)
    assert report["passed"]
    assert all(report["checks"].values())
    assert report["exact_qft3"]["capability"]["classification"] == (
        "APPROXIMATION_REQUIRED"
    )
    assert report["exact_qft3"]["native_search_target_created"] is False
    assert report["aqft3"]["reference"]["omitted_operations"][0]["angle_pi"] == "1/4"

    expected_names = {
        "summary.json",
        "summary.md",
        "exact_qft3.svg",
        "aqft3.svg",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_names
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    assert summary["artifacts"]["exact_qft3_diagram"] == "exact_qft3.svg"

    exact_svg = (tmp_path / "exact_qft3.svg").read_text(encoding="utf-8")
    approximate_svg = (tmp_path / "aqft3.svg").read_text(encoding="utf-8")
    assert "SDK-neutral high-level reference; not a native-search witness" in exact_svg
    assert "All exact high-level QFT-3 operations are shown" in exact_svg
    assert "Omitted by declared AQFT approximation" in approximate_svg
    assert "ControlledPhase(1/4 pi; q2,q0)" in approximate_svg


def test_qft_benchmark_cli_exit_code_tracks_acceptance_gate(tmp_path, capsys):
    assert main(["--artifacts-dir", str(tmp_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["passed"]
    assert printed["checks"]["native_search_grammar_is_unchanged"]
