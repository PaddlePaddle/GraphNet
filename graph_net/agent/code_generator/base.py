"""Base code generator interface"""

from abc import ABC, abstractmethod
from pathlib import Path

from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata


class BaseCodeGenerator(ABC):
    """Base class for code generators"""

    @abstractmethod
    def generate(
        self,
        model_dir: Path,
        model_metadata: ModelMetadata,
        output_dir: Path,
    ) -> Path:
        """
        Generate run_model.py extraction script

        Args:
            model_dir: Path to model directory
            model_metadata: Model metadata extracted from configuration
            output_dir: Output directory for generated script

        Returns:
            Path to generated script file

        Raises:
            CodeGenError: If code generation fails
        """
        pass
