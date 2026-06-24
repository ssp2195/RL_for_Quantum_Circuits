from typing import List

from circuit.gate import Gate
from search.node import SearchNode
from search.action import Action


def expand_node(node: SearchNode, actions: List[Action], policy) -> List[SearchNode]:
    """
    Expands a node into child nodes using given actions.
    """

    children = []

    for action in actions:
        child = _apply_action(node, action, policy)

        if child is not None:
            children.append(child)

    return children


# =========================================================
# Internal helpers
# =========================================================

def _apply_action(node: SearchNode, action: Action, policy):
    state = node.state.copy()

    gate = Gate(action.gate_type, action.qubits)

    success = state.apply_gate(gate)

    if not success:
        return None

    if node.action is not None:
        if action == node.action:
            return None

    # Avoid H followed by H (H^2 = I)
    if node.action is not None:
        if node.action.gate_type == action.gate_type == GateType.H:
            return None

    # simple heuristic priority (can be replaced by RL later)
    priority = _compute_priority(state, policy)

    return SearchNode(
        priority=priority,
        state=state,
        parent=node,
        action=action,
    )


def _compute_priority(state, policy):
    """
    RL-based priority:
    Higher Q → better → lower priority value
    """
    return policy.score_state(state)
