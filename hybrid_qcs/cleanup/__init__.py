"""Theorem-guarded full-bank cleanup synthesis.

This is a structured dynamic-protocol backend, not a replacement for the native
unitary search. Local coherent blocks retain the authoritative HybridState.
"""
from .contract import CleanupProblem, Consumer, Hardware, Limits, Layout
from .ir import Instruction, Protocol
from .verify import verify_protocol, verify_workspace_certificate

__all__ = ['CleanupProblem', 'Consumer', 'Hardware', 'Limits', 'Layout',
           'Instruction', 'Protocol', 'verify_protocol', 'verify_workspace_certificate']
