# Article V1 ten-minute protocol amendment

The synthesis architecture is unchanged. The learned action remains selection
of one persistent open frontier record; each selected record is expanded over
every resource-feasible native Clifford+T continuation. Symbolic semantics,
canonicalization, Pareto pruning, resource limits, and fresh independent final
certification remain shared by every scheduler.

The earlier 8,192- and 100,000-expansion horizons were provisional engineering
settings, not theorem requirements. The amended hard horizon is selected using
train/validation process-CPU feasibility only: the largest preregistered cap
whose 95th-percentile CPU time is at most 540 seconds, whose maximum is at most
600 seconds, whose exact feature index stays within 100 MiB, and whose runs
contain no timeout or correctness failure.

Primary SARSA training uses a fixed total number of expansions over easy and
medium train targets. Hard targets are excluded from the recommended checkpoint
and reported as generalization/stress evaluation. Evaluation executes one
trajectory at the frozen maximum horizon and derives the labelled
`fixed-horizon anytime budget-success curve` from its first certified hit.
Because remaining budget is a feature, this curve is not represented as a set
of independently trained/rerun smaller-horizon policies.

All seven primary schedulers receive identical expansion horizons. The
equal-process-CPU comparison is supplementary and remains separate from the
primary equal-expansion analysis.

Exact search remains exponential. The ten-minute horizon limits completeness
under an external budget; it does not change the exactness of a returned
circuit and does not establish polynomial or large-scale synthesis.
