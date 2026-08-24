"""Hard-deadline command-line runner for the hybrid QCS campaign."""
from __future__ import annotations

import argparse

from .runner_campaign import execute_campaign
from .runner_report import write_campaign_report


def run(args: argparse.Namespace) -> int:
    return write_campaign_report(args, execute_campaign(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/hybrid-30min")
    parser.add_argument("--deadline-seconds", type=float, default=1800.0)
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--seeds", default="11,19,23,31,47")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--train-expansion-cap", type=int, default=2048)
    parser.add_argument("--eval-expansion-cap", type=int, default=8192)
    parser.add_argument("--toffoli-expansion-cap", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
