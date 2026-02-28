"""Base sample verifier interface"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseSampleVerifier(ABC):
    """Base class for sample verifiers"""

    @abstractmethod
    def verify(self, sample_dir: Path) -> bool:
        """
        Verify sample validity

        Args:
            sample_dir: Path to sample directory

        Returns:
            True if sample is valid, False otherwise

        Raises:
            VerificationError: If verification process fails
        """
        pass
