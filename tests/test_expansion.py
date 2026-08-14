from dataclasses import dataclass, field

from enums import GateType
from search.action import Action
from search.action_space import generate_actions
from search.expansion import expand_node
from search.node import SearchNode


@dataclass
class PermissiveState:
    """Minimal state double for testing expansion legality, not algebra."""

    t_count: int = 0
    depth: int = 0
    num_gates: int = 0
    applied: list = field(default_factory=list)

    def copy(self):
        return PermissiveState(
            t_count=self.t_count,
            depth=self.depth,
            num_gates=self.num_gates,
            applied=list(self.applied),
        )

    def apply_gate(self, gate):
        self.applied.append(gate)
        self.num_gates += 1
        self.depth += 1
        if gate.gate_type in {GateType.T, GateType.TDG}:
            self.t_count += 1
        return True


def test_default_action_space_includes_clifford_and_t_inverses():
    actions = generate_actions(2)

    assert len(actions) == 2 * 5 + 2
    for qubit in range(2):
        assert Action(GateType.H, (qubit,)) in actions
        assert Action(GateType.S, (qubit,)) in actions
        assert Action(GateType.SDG, (qubit,)) in actions
        assert Action(GateType.T, (qubit,)) in actions
        assert Action(GateType.TDG, (qubit,)) in actions
    assert Action(GateType.CNOT, (0, 1)) in actions
    assert Action(GateType.CNOT, (1, 0)) in actions


def test_expansion_keeps_every_legal_repeated_gate_continuation():
    repeated_t = Action(GateType.T, (0,))
    repeated_s = Action(GateType.S, (0,))
    repeated_cnot = Action(GateType.CNOT, (0, 1))
    actions = [repeated_t, repeated_s, repeated_cnot]
    parent = SearchNode(
        priority=0.0,
        state=PermissiveState(),
        action=repeated_t,
    )

    children = expand_node(parent, actions)

    assert [child.action for child in children] == actions
    assert all(child.parent is parent for child in children)
    assert all(child.state.applied[-1].gate_type == child.action.gate_type for child in children)


def test_node_reconstructs_a_concrete_witness_in_root_to_leaf_order():
    root = SearchNode(priority=0.0, state=PermissiveState())
    first_action = Action(GateType.H, (0,))
    second_action = Action(GateType.T, (0,))
    first = SearchNode(
        priority=1.0,
        state=PermissiveState(),
        parent=root,
        action=first_action,
    )
    leaf = SearchNode(
        priority=2.0,
        state=PermissiveState(),
        parent=first,
        action=second_action,
    )

    assert list(leaf.iter_path()) == [root, first, leaf]
    assert leaf.reconstruct_actions() == [first_action, second_action]
    assert leaf.witness_actions() == [first_action, second_action]
