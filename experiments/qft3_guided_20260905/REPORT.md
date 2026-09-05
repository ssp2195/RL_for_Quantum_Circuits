# Decomposition-guided exact QFT-3 mitigation

## Interpretation of the earlier negative result

The generic four-qubit native search did not rediscover QFT-3 within the bounded probe. This is a search-budget failure, not a semantic or certification failure. The independently reconstructed reference witness and the decomposition-guided circuit both satisfy the three-logical-qubit, one-clean-ancilla isometry contract.

## Implemented mitigation

The implementation recognizes the analytical forward QFT matrix, derives the standard seven-block QFT factorization from the declared logical wire ordering, lowers every block into the unchanged native `H`, `S`, `SDG`, `T`, `TDG`, and `CNOT` grammar, and independently replays and certifies the resulting DAG.

The controlled-`T` block uses a relative-phase AND compute circuit `R`, applies `T` to the clean ancilla, and executes `R^dagger`. Because the same relative phases occur on both sides of the diagonal ancilla phase, they cancel exactly on the clean-input subspace.

## Final bounded qualification

| Quantity | Result |
|---|---:|
| Guided generation certified | True |
| Native gates | 35 |
| `T`/`TDG` gates | 15 |
| CNOT gates | 13 |
| Depth | 26 |
| Projective isometry error | 1.244e-15 |
| Exact isometry error | 1.383e-15 |
| Clean-ancilla leakage | 3.830e-33 |
| Guided generation wall time | 0.003156 s |
| Additional training episodes | 0 |

The original certified reference witness used 47 native gates, 21 `T`/`TDG` gates, 19 CNOTs, and depth 34. The mitigation therefore removes 12 native gates, six non-Clifford gates, six CNOTs, and eight depth levels.

The matched training protocol remained marginal: the ancilla-specific increment was 0.183232 s over a 2.118106 s reference, for a total/reference ratio of 1.0865x. The QFT macro path itself performs no training.

## Unrestricted baseline retained

The unrestricted deferred SARSA/LinUCB probe remains negative and is not relabelled as success: it stopped by `wall_limit` after 149 outer allocations and 581 exact edge attempts in 1.516610 s. This distinction prevents the structured compiler result from being presented as unrestricted native discovery.

## Regression status

`pytest`: 52 passed.
