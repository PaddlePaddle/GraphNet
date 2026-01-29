"""Template-based code generator implementation"""

from pathlib import Path
from typing import Optional

from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata
from graph_net.agent.code_generator.base import BaseCodeGenerator
from graph_net.agent.utils.exceptions import CodeGenError

# Constants for safe vocab size calculation
DEFAULT_VOCAB_SIZE = 30522
MIN_SAFE_VOCAB_SIZE = 100
FIXED_SAFE_LIMIT = 25000  # For very large vocabularies or specific model types
LARGE_VOCAB_THRESHOLD = 100000
MEDIUM_VOCAB_THRESHOLD = 50000
SMALL_VOCAB_THRESHOLD = 10000
LARGE_VOCAB_RATIO = 0.7
MEDIUM_VOCAB_RATIO = 0.8
SMALL_VOCAB_RATIO = 0.9


class TemplateCodeGenerator(BaseCodeGenerator):
    """Code generator using Jinja2 template"""

    def __init__(self, template_path: Optional[str] = None):
        """
        Args:
            template_path: Path to Jinja2 template file (optional)
        """
        self.template_path = template_path
        self._template = None

    def generate(
        self,
        model_dir: Path,
        model_metadata: ModelMetadata,
        output_dir: Path,
    ) -> Path:
        """
        Generate run_model.py extraction script using template

        Args:
            model_dir: Path to model directory
            model_metadata: Model metadata extracted from configuration
            output_dir: Output directory for generated script

        Returns:
            Path to generated script file
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            code = self._generate_code(model_dir, model_metadata)

            script_path = output_dir / "run_model.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            return script_path
        except Exception as e:
            raise CodeGenError(f"Failed to generate code: {e}") from e

    def _generate_code(self, model_dir: Path, model_metadata: ModelMetadata) -> str:
        """Generate complete extraction script code string"""
        # Generate model loading code
        load_code = self._generate_model_loader(model_dir, model_metadata)

        # Generate input construction code
        input_code = self._generate_input_code(model_metadata)

        # Generate main code
        code = f"""import torch
try:
    from transformers import AutoModel
except ImportError:
    raise ImportError("transformers is required. Install with: pip install transformers")

import graph_net

def main():
    # Load model
{self._indent(load_code, 4)}
    
    # Prepare inputs
{self._indent(input_code, 4)}
    
    # Extract graph
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Move inputs to same device as model
    inputs = {{k: v.to(device) for k, v in inputs.items()}}
    
    wrapped = graph_net.torch.extract(name="{model_metadata.model_id}", dynamic=True)(model).eval()
    
    with torch.no_grad():
        wrapped(**inputs)

if __name__ == "__main__":
    main()
"""
        return code

    def _generate_model_loader(
        self, model_dir: Path, model_metadata: ModelMetadata
    ) -> str:
        """Generate model loading code based on model type"""
        model_path = str(model_dir).replace("\\", "/")

        if model_metadata.model_type in ["bert", "gpt", "t5", "roberta"]:
            return f'model = AutoModel.from_pretrained("{model_path}")'
        elif model_metadata.model_type in ["resnet", "vgg", "densenet"]:
            return f"model = torchvision.models.{model_metadata.model_type}(pretrained=True)"
        else:
            # Generic loading
            return f'model = AutoModel.from_pretrained("{model_path}")'

    def _generate_input_code(self, model_metadata: ModelMetadata) -> str:
        """Generate input tensor construction code based on model metadata"""
        lines = ["inputs = {}"]

        for name, shape in model_metadata.input_shapes.items():
            dtype = model_metadata.input_dtypes.get(name, "int64")
            torch_dtype = self._get_torch_dtype(dtype)
            shape_tuple = f"({', '.join(map(str, shape))})"

            if dtype == "int64":
                if "input_ids" in name.lower():
                    safe_vocab_size = self._calculate_safe_vocab_size(model_metadata)
                    lines.append(
                        f'inputs["{name}"] = torch.randint(0, {safe_vocab_size}, {shape_tuple}, dtype={torch_dtype})'
                    )
                else:
                    lines.append(
                        f'inputs["{name}"] = torch.ones({shape_tuple}, dtype={torch_dtype})'
                    )
            else:
                lines.append(
                    f'inputs["{name}"] = torch.randn({shape_tuple}, dtype={torch_dtype})'
                )

        return "\n".join(lines)

    def _get_torch_dtype(self, dtype: str) -> str:
        """Convert dtype string to torch dtype"""
        if dtype == "int64":
            return "torch.int64"
        elif dtype == "float32":
            return "torch.float32"
        else:
            return f"torch.{dtype}"

    def _calculate_safe_vocab_size(self, model_metadata: ModelMetadata) -> int:
        """Calculate safe vocabulary size for input generation"""
        if model_metadata.embedding_size:
            return max(MIN_SAFE_VOCAB_SIZE, model_metadata.embedding_size - 1)

        vocab_size = model_metadata.vocab_size or DEFAULT_VOCAB_SIZE
        model_type = (model_metadata.model_type or "").lower()

        # Model-type-specific limits
        if self._is_large_vocab_model_type(model_type):
            return FIXED_SAFE_LIMIT

        # Size-based strategy
        if vocab_size > LARGE_VOCAB_THRESHOLD:
            return FIXED_SAFE_LIMIT
        elif vocab_size > MEDIUM_VOCAB_THRESHOLD:
            return max(MIN_SAFE_VOCAB_SIZE, int(vocab_size * LARGE_VOCAB_RATIO))
        elif vocab_size > SMALL_VOCAB_THRESHOLD:
            return max(MIN_SAFE_VOCAB_SIZE, int(vocab_size * MEDIUM_VOCAB_RATIO))
        else:
            return max(MIN_SAFE_VOCAB_SIZE, int(vocab_size * SMALL_VOCAB_RATIO))

    def _is_large_vocab_model_type(self, model_type: str) -> bool:
        """Check if model type typically has large vocabulary but small embedding"""
        return (
            "xlm-roberta" in model_type
            or "xlm_roberta" in model_type
            or "roberta" in model_type
        )

    def _indent(self, text: str, spaces: int) -> str:
        """Indent text by specified spaces"""
        indent = " " * spaces
        return "\n".join(indent + line for line in text.split("\n"))
