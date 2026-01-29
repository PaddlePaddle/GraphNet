"""Basic sample verifier implementation"""

from pathlib import Path

from graph_net.agent.sample_verifier.base import BaseSampleVerifier
from graph_net.agent.utils.exceptions import VerificationError


class BasicSampleVerifier(BaseSampleVerifier):
    """Basic verifier that checks file existence and basic structure"""

    def __init__(self):
        """Initialize basic verifier"""
        self.required_files = [
            "model.py",
            "graph_net.json",
            "input_meta.py",
            "weight_meta.py",
        ]

    def verify(self, sample_dir: Path) -> bool:
        """
        Verify sample validity

        Args:
            sample_dir: Path to sample directory

        Returns:
            True if sample is valid, False otherwise
        """
        try:
            # Check required files exist
            for filename in self.required_files:
                file_path = sample_dir / filename
                if not file_path.exists():
                    return False

            # Check graph_net.json is valid JSON
            json_path = sample_dir / "graph_net.json"
            try:
                import json

                with open(json_path, "r") as f:
                    json.load(f)
            except (json.JSONDecodeError, IOError):
                return False

            return True
        except Exception as e:
            raise VerificationError(f"Verification failed: {e}") from e
