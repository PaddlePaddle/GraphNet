from abc import ABC, abstractmethod
from typing import Any, Dict

from ..backends.base_backend import BaseBackend


class BenchmarkResult:
    """Aggregated result of a benchmark run."""

    def __init__(self, outputs: Any, metrics: Dict[str, Any]):
        self.outputs = outputs
        self.metrics = metrics


class BaseRunner(ABC):
    """Base class for all runners."""

    @abstractmethod
    def load_backend(
        self, backend_spec: Dict[str, Any], model_path: str
    ) -> BaseBackend:
        """Load and initialize a backend instance."""
        pass

    @abstractmethod
    def cleanup_backend(self, backend: BaseBackend) -> None:
        """Cleanup backend resources."""
        pass

    @abstractmethod
    def run_benchmark(
        self, backend_spec: Dict[str, Any], model_path: str
    ) -> BenchmarkResult:
        """Run the full benchmark (warmup + trials) and return aggregated result."""
        pass
