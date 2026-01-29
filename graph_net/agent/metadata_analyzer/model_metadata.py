"""Model metadata data structure"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ModelMetadata:
    """Model metadata containing input shapes, dtypes, and model type"""

    model_id: str
    input_shapes: Dict[str, List[int]]
    input_dtypes: Dict[str, str]
    model_type: Optional[str] = None  # e.g., "bert", "resnet", "gpt"
    vocab_size: Optional[int] = None  # Vocabulary size for tokenizer models
    embedding_size: Optional[
        int
    ] = None  # Actual embedding layer size (from model weights)

    def __post_init__(self):
        """Validate metadata"""
        if not self.input_shapes:
            raise ValueError("input_shapes cannot be empty")
        if not self.input_dtypes:
            raise ValueError("input_dtypes cannot be empty")

        # Ensure shapes and dtypes have matching keys
        if set(self.input_shapes.keys()) != set(self.input_dtypes.keys()):
            raise ValueError("input_shapes and input_dtypes must have matching keys")
