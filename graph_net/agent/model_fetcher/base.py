"""Base model fetcher interface"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseModelFetcher(ABC):
    """Base class for model fetchers"""

    @abstractmethod
    def download(self, model_id: str) -> Path:
        """
        Download model to local directory

        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")

        Returns:
            Path to local model directory

        Raises:
            ModelFetchError: If download fails
        """
        pass
