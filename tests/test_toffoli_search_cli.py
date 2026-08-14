"""Fast CLI failure-path coverage for the Stage 3 learned-search runner."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path

from toffoli_search import main


def test_toffoli_search_cli_returns_nonzero_without_a_learned_witness(
    tmp_path: Path,
) -> None:
    """A deliberately too-small horizon must fail honestly and save artifacts.

    This is intentionally a short CLI regression, not a replacement for the
    real seeded ``--train`` benchmark.  It exercises the public exit-status
    and presentation boundary: an unsuccessful learned run may write its
    report, but it must neither claim success nor draw a substituted reference
    circuit.
    """

    output_dir = tmp_path / "toffoli-search-failed"
    with redirect_stdout(StringIO()):
        exit_code = main(
            [
                "--artifacts-dir",
                str(output_dir),
                "--train",
                "--episodes",
                "0",
                "--training-max-steps",
                "1",
                "--evaluation-max-steps",
                "1",
                "--random-reproducibility-max-steps",
                "1",
            ]
        )

    assert exit_code == 1
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["correct"] is False
    assert summary["learned_witness"]["certified"] is False
    assert summary["learned_witness"]["operations"] == []
    assert summary["fifo"]["seed"] == 24
    assert summary["fifo"]["step_cap"] == 1
    assert "archive_record_count" in summary["fifo"]["search_metrics"]
    assert set(summary["random_seed_checks"]) == {"23", "24", "25"}
    assert all(check["reproducible"] for check in summary["random_seed_checks"].values())
    diagram = (output_dir / "circuit.svg").read_text(encoding="utf-8")
    assert "No certified learned Toffoli witness" in diagram
    assert "Actual gate sequence reconstructed" not in diagram
