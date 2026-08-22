# Article V1 ten-minute optimization evidence

Status: engineering diagnostic on the uncommitted `ten-minute-protocol`
worktree. This file is not a clean-source publication qualification artifact.

The comparison uses the fixed pilot `train/hard/3q` workload, training seed 24,
the unchanged scientific horizon of 8192, and an engineering stop at the stated
cap. The baseline comes from the reviewed clean-source artifact
`article-v1-feature-index-v2-clean-2ab3361-20260818/end_to_end_scaling.csv`.

| Expansion cap | Clean baseline (s) | Current exact implementation (s) | Speedup | Peak frontier parity | Peak group parity | Index-memory parity |
|---:|---:|---:|---:|:---:|:---:|:---:|
| 512 | 59.0407569 | 45.2420840 | 1.305x | yes (6763) | yes (1237) | yes (4598120 B) |
| 1024 | 218.0201876 | 106.0011734 | 2.057x | yes (13142) | yes (2044) | yes (8508576 B) |

The exact optimized/reference trace qualification also passes independently at
caps 8, 16, 32, and 64. It compares selected persistent record IDs, rewards,
TD errors, final weights, certification and witness data, frontier/archive
state, and deterministic counters. Frontier enumeration retains the reviewed
priority-then-record-ID order and caches that view only within an unchanged
frontier revision.

Both measured expansion points exceed the plan's 1.25x end-to-end speedup gate.
The 2048 cap remains a calibration candidate; it is not selected or frozen by
this diagnostic. Selection still requires the preregistered train/validation
Q95 CPU, hard-limit, timeout, memory, and correctness gates.
