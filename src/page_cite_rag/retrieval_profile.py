"""One explicit local hybrid retrieval profile shared by product and evaluation.

The values are starting engineering choices from prior local runs, not universal
optima: 20 candidates per branch, RRF k=60, at most 10 final source blocks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalProfile:
    version: str = "formal-hybrid-v1"
    candidates_per_branch: int = 20
    rrf_constant: int = 60
    final_k: int = 10
    neighbor_blocks: int = 1

    def as_dict(self) -> dict[str, int | str]:
        return {
            "version": self.version,
            "candidates_per_branch": self.candidates_per_branch,
            "rrf_constant": self.rrf_constant,
            "final_k": self.final_k,
            "neighbor_blocks": self.neighbor_blocks,
        }


FORMAL_RETRIEVAL = RetrievalProfile()
