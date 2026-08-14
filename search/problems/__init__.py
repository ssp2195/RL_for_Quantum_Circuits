"""Deterministic constrained synthesis problems for the shared frontier."""

from search.problems.base import SearchProblem
from search.problems.native import NativeGateSearchProblem
from search.problems.toffoli_parity import (
    CNOT_BASIS_DISTANCE_TO_IDENTITY,
    REQUIRED_PHASE_TERMS,
    ToffoliParityNetworkProblem,
    ToffoliParityProgress,
    ToffoliProblemCanonicalizer,
    ToffoliStage,
    analyze_toffoli_prefix,
)

__all__ = [
    "CNOT_BASIS_DISTANCE_TO_IDENTITY",
    "NativeGateSearchProblem",
    "REQUIRED_PHASE_TERMS",
    "SearchProblem",
    "ToffoliParityNetworkProblem",
    "ToffoliParityProgress",
    "ToffoliProblemCanonicalizer",
    "ToffoliStage",
    "analyze_toffoli_prefix",
]
