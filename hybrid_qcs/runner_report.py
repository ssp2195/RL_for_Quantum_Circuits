"""Artifact generation for the bounded hybrid-QCS qualification campaign."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .runner_artifacts import write_machine_artifacts
from .runner_support import _component_timing
from .runner_summary import write_human_summary


def write_campaign_report(args: argparse.Namespace, context: dict[str, Any]) -> int:
    del args  # CLI values are already captured in the campaign context.
    output = Path(context["output"])
    output.mkdir(parents=True, exist_ok=True)
    enriched = dict(context)
    enriched["component_timing"] = _component_timing(
        list(context["seed_results"]),
        list(context["baselines"]),
        list(context["scaling"]),
    )
    write_machine_artifacts(output, enriched)
    return write_human_summary(output, enriched)


__all__ = ["write_campaign_report"]
