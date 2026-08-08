import heapq
from typing import Dict, Tuple

from canonical.canonicalizer import Canonicalizer
from search.node import SearchNode


class Frontier:
    """
    Priority queue with:
        - identity deduplication
        - dominance pruning
    """

    def __init__(self):
        self.heap = []

        # identity_hash -> best resource tuple seen
        self.visited: Dict[str, Tuple[int, int, int]] = {}

        self.canonicalizer = Canonicalizer()

    # =========================================================
    # Insert with dominance pruning
    # =========================================================

    def push(self, node: SearchNode) -> bool:
        """
        Returns True if inserted, False if pruned
        """

        state = node.state

        identity = self.canonicalizer.identity_hash(state)
        resource = self.canonicalizer._resource_canonical_form(state)

        # ---------- Check dominance ----------

        if identity in self.visited:
            best_resource = self.visited[identity]

            if self._dominates(best_resource, resource):
                # Existing is better → prune new
                return False

            if self._dominates(resource, best_resource):
                # New is better → replace
                self.visited[identity] = resource
            else:
                # Neither dominates → keep both (rare but important)
                pass
        else:
            self.visited[identity] = resource

        # ---------- Push to heap ----------
        heapq.heappush(self.heap, node)
        return True

    # =========================================================
    # Pop
    # =========================================================

    def pop(self) -> SearchNode:
        return heapq.heappop(self.heap)

    def remove(self, node: SearchNode) -> bool:
        """
        Remove a node by identity. Returns True if removed.
        """
        for i, n in enumerate(self.heap):
            if n is node:
                del self.heap[i]
                heapq.heapify(self.heap)
                return True
        return False

    def is_empty(self) -> bool:
        return len(self.heap) == 0

    # =========================================================
    # Dominance Logic
    # =========================================================

    def _dominates(self, r1, r2) -> bool:
        """
        r1 dominates r2 if:
            r1 <= r2 in all dimensions
            and strictly < in at least one
        """
        leq = all(a <= b for a, b in zip(r1, r2))
        strictly_less = any(a < b for a, b in zip(r1, r2))
        return leq and strictly_less
