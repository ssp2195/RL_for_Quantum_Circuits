# Clean-ancilla hierarchical QCS qualification

## Scope

The implementation supports a fixed physical register partitioned into logical qubits, clean |0> ancillas, and optionally borrowed ancillas. Correctness is contract-relative isometry equality; clean workspace must be returned to |0>, and borrowed workspace is required to undergo the identity operation on an arbitrary input state.

## Training cost

- Existing matched hierarchical training: 2.134339 s
- Existing training plus ancilla fine-tuning: 2.308604 s
- Total/reference ratio: 1.082x

The ancilla policies are warm-started from the existing mixed-gate linear models and receive only a short staged fine-tuning pass. No joint neural training or graph encoder is introduced.

## Held-out clean-ancilla results

| Target | Certified | Wall (s) | Allocations | Exact edges | Peak frontier |
|---|---:|---:|---:|---:|---:|
| heldout-clean-h-t-echo-1q | True | 0.001870 | 2 | 6 | 7 |
| heldout-clean-s-t-product-2q | True | 0.002029 | 2 | 5 | 6 |
| heldout-clean-mixed-product-3q | True | 0.004378 | 3 | 9 | 10 |

## QFT-3 with one clean ancilla

The independently constructed 47-gate native witness is certified: **True**. Its projective isometry error is 1.124e-15 and its clean-ancilla leakage is 2.151e-32.

The unrestricted hierarchical search result is **wall_limit** after 150 outer allocations and 585 exact continuation attempts. This bounded probe is reported honestly; witness certification establishes representability and contract correctness, while an unsuccessful search does not establish unrestricted synthesis at this depth.

## Claim boundary

Archive pruning remains based on the strengthened full-register projective key. This is sound but incomplete for clean-ancilla equivalence: full-unitary equality implies isometry equality, but the converse need not hold. Terminal acceptance no longer requires full symbolic-key equality and is decided independently by the ancilla isometry certifier.
