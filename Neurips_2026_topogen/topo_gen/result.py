"""Result types for graph generation attempts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

import networkx as nx


class FailureReason(Enum):
    H0_MERGE_FAILURE        = auto()  # Step 3: no valid merge pair for an H0 death event
    NO_CYCLE_CANDIDATE      = auto()  # Step 4: no valid intra-component pair for an H1 birth
    NO_NEW_VERTEX_PAIR      = auto()  # Step 3/4: stalemate — all remaining merges need a new-vertex but none available
    CAPACITY_EXHAUSTED      = auto()  # Step 3/4: degree constraint — no valid pair with remaining capacity


@dataclass
class Success:
    H         : nx.Graph
    node_time : dict
    edge_time : dict

    @property
    def ok(self) -> bool:
        return True


@dataclass
class Failure:
    reason    : FailureReason
    step      : int   # 3 or 4
    timestep  : int   # k at which the failure occurred
    detail    : str

    @property
    def ok(self) -> bool:
        return False


GenerationResult = Union[Success, Failure]
