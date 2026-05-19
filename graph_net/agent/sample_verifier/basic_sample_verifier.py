"""Basic sample verifier implementation"""

import json
from pathlib import Path

from graph_net.agent.sample_verifier.base import BaseSampleVerifier
from graph_net.agent.utils.exceptions import (
    GraphExtractionErrorCategory,
    SampleVerificationError,
)


class BasicSampleVerifier(BaseSampleVerifier):
    """Basic verifier that checks file existence and basic structure.

    Supports both single-graph and multi-subgraph (subgraph_0/, subgraph_1/, …) layouts.
    """

    def __init__(self):
        self.required_files = [
            "model.py",
            "graph_net.json",
            "input_meta.py",
            "weight_meta.py",
        ]

    def verify(self, sample_dir: Path) -> bool:
        try:
            subgraph_dirs = sorted(sample_dir.glob("subgraph_*/"))
            targets = subgraph_dirs if subgraph_dirs else [sample_dir]

            for target in targets:
                for filename in self.required_files:
                    if not (target / filename).exists():
                        return False
                try:
                    with open(target / "graph_net.json", "r") as f:
                        json.load(f)
                except (json.JSONDecodeError, IOError):
                    return False

            return True
        except Exception as e:
            raise SampleVerificationError(
                f"Verification failed: {e}",
                error_category=GraphExtractionErrorCategory.SAMPLE_INCOMPLETE,
            ) from e
