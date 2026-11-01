"""One hybrid profile shared by the product and evaluation paths.

Each lexical/dense branch returns 50 candidates. The fused result can contain up to
100 source groups; exact token admission, rather than a fixed ten-group cutoff,
determines how much evidence reaches the writer.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_CONTEXT_GROUPS = 100


@dataclass(frozen=True)
class RetrievalProfile:
    version: str = "formal-hybrid-v2"
    candidates_per_branch: int = 50
    rrf_constant: int = 60
    final_k: int = MAX_CONTEXT_GROUPS
    neighbor_blocks: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "version": self.version,
            "candidates_per_branch": self.candidates_per_branch,
            "rrf_constant": self.rrf_constant,
            "final_k": self.final_k,
            "neighbor_blocks": self.neighbor_blocks,
        }


FORMAL_RETRIEVAL = RetrievalProfile()
