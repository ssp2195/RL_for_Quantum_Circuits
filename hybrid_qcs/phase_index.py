"""An exact indexed frontier heap; the policy panel never bounds the frontier.

Insertion/deletion costs O(log F). Reading the K smallest records visits at most
2K+1 heap locations and costs O(K log K), without lazy-deletion scans. Permanent
record IDs are unrelated to storage positions. This structure contains no model.
"""
from __future__ import annotations
import heapq


class CandidateIndex:
    def __init__(self):
        self.heap = []
        self.position = {}

    def __len__(self):
        return len(self.heap)

    def _swap(self, a, b):
        self.heap[a], self.heap[b] = self.heap[b], self.heap[a]
        self.position[self.heap[a][1]] = a
        self.position[self.heap[b][1]] = b

    def _up(self, i):
        while i:
            parent = (i-1)//2
            if self.heap[parent] <= self.heap[i]:
                break
            self._swap(i, parent)
            i = parent
        return i

    def _down(self, i):
        while 2*i+1 < len(self.heap):
            child = 2*i+1
            if child+1 < len(self.heap) and self.heap[child+1] < self.heap[child]:
                child += 1
            if self.heap[i] <= self.heap[child]:
                break
            self._swap(i, child)
            i = child

    def add(self, priority, rid):
        if rid in self.position:
            raise ValueError('duplicate persistent frontier record')
        self.position[rid] = len(self.heap)
        self.heap.append((priority, rid))
        self._up(len(self.heap)-1)

    def discard(self, rid):
        if rid not in self.position:
            return
        i = self.position.pop(rid)
        last = self.heap.pop()
        if i == len(self.heap):
            return
        self.heap[i] = last
        self.position[last[1]] = i
        self._down(self._up(i))

    def smallest(self, k):
        if k < 0:
            raise ValueError('negative panel size')
        if not self.heap or not k:
            return ()
        pending = [(self.heap[0], 0)]
        result = []
        while pending and len(result) < k:
            (_, rid), index = heapq.heappop(pending)
            result.append(rid)
            for child in (2*index+1, 2*index+2):
                if child < len(self.heap):
                    heapq.heappush(pending, (self.heap[child], child))
        return tuple(result)

    def minimum(self):
        return self.heap[0] if self.heap else None
