"""Base graph extractor interface"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseGraphExtractor(ABC):
    """Base class for computation graph extractors"""

    @abstractmethod
    def extract(self, code_path: Path, model_id: str) -> Path:
        """
        Execute script and extract computation graph

        Args:
            code_path: Path to run_model.py script
            model_id: Model ID for output directory naming

        Returns:
            Path to extracted sample directory

        Raises:
            ExtractionError: If extraction fails
        """
        pass
