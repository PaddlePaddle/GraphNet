"""Base metadata analyzer interface"""

from abc import ABC, abstractmethod
from pathlib import Path

from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata


class BaseMetadataAnalyzer(ABC):
    """Base class for model metadata analyzers"""

    @abstractmethod
    def analyze(self, model_dir: Path) -> ModelMetadata:
        """
        Analyze model configuration and extract metadata

        Args:
            model_dir: Path to model directory

        Returns:
            ModelMetadata object containing model information

        Raises:
            AnalysisError: If analysis fails
        """
        pass
