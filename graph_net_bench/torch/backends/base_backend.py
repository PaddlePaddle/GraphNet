from abc import ABC, abstractmethod
from typing import Any, Dict


class ExecuteResult:
    """Result of a single backend execution."""

    def __init__(self, outputs: Any, metrics: Dict[str, float]):
        self.outputs = outputs
        self.metrics = metrics


class BaseBackend(ABC):
    """Base class for all backends."""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config

    @abstractmethod
    def execute(self) -> ExecuteResult:
        """Execute a single inference.

        Returns:
            ExecuteResult containing outputs and metrics.
            Timing in metrics must include device synchronization.
        """
        pass

    def warmup(self, num_warmup: int) -> None:
        """Warmup runs before benchmark.

        Default implementation executes num_warmup times.
        Override for custom warmup logic.
        """
        for _ in range(num_warmup):
            self.execute()

    def cleanup(self) -> None:
        """Release resources. Override if needed."""
        pass
