"""Template-based code generator implementation"""

from pathlib import Path
from typing import Optional

from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata
from graph_net.agent.code_generator.base import BaseCodeGenerator
from graph_net.agent.utils.exceptions import CodeGenerationError

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
            raise CodeGenerationError(f"Failed to generate code: {e}") from e

    @staticmethod
    def _model_short_name(model_id: str) -> str:
        """Return 'org_model' name (replace '/' with '_')"""
        return model_id.replace("/", "_")

    def _generate_code(self, model_dir: Path, model_metadata: ModelMetadata) -> str:
        """Generate complete extraction script code string."""
        if model_metadata.architecture_type == "diffusion":
            return self._generate_diffusion_code(model_dir, model_metadata)
        return self._generate_standard_code(model_dir, model_metadata)

    def _generate_standard_code(
        self, model_dir: Path, model_metadata: ModelMetadata
    ) -> str:
        """Generate standard (transformers-based) extraction script."""
        load_code = self._generate_model_loader(model_dir, model_metadata)
        input_code = self._generate_input_code(model_metadata)
        short_name = self._model_short_name(model_metadata.model_id)

        code = f"""import torch
import graph_net

{load_code}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
{input_code}

graph_net.torch.extract(name="{short_name}", dynamic=False)(model).eval()(**inputs)
"""
        return code

    def _generate_diffusion_code(
        self, model_dir: Path, model_metadata: ModelMetadata
    ) -> str:
        """Generate extraction script for diffusion models (diffusers UNet)."""
        load_code = self._generate_model_loader(model_dir, model_metadata)
        input_code = self._generate_input_code(model_metadata)
        short_name = self._model_short_name(model_metadata.model_id)

        code = f"""import torch
import graph_net

{load_code}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
{input_code}

sample = inputs["sample"]
timestep = inputs["timestep"]
encoder_hidden_states = inputs["encoder_hidden_states"]
graph_net.torch.extract(name="{short_name}", dynamic=False)(model).eval()(sample, timestep, encoder_hidden_states)
"""
        return code

    def _generate_model_loader(
        self, model_dir: Path, model_metadata: ModelMetadata
    ) -> str:
        """Generate model loading code based on architecture type."""
        model_path = str(model_dir).replace("\\", "/")
        arch = model_metadata.architecture_type

        if arch == "seq2seq":
            return (
                f"from transformers import AutoConfig, AutoModelForSeq2SeqLM\n"
                f'_config = AutoConfig.from_pretrained("{model_path}", trust_remote_code=True)\n'
                f"model = AutoModelForSeq2SeqLM.from_config(_config)"
            )
        elif arch == "diffusion":
            return (
                f"from diffusers import UNet2DConditionModel\n"
                f'_config = UNet2DConditionModel.load_config("{model_path}")\n'
                f"model = UNet2DConditionModel.from_config(_config)"
            )
        else:
            # text, moe, vision, multimodal, audio, None → AutoModel
            # If model_type is not present in config.json (e.g. prajjwal1/bert-tiny),
            # inject the inferred model_type so AutoConfig can resolve the class.
            model_type = model_metadata.model_type
            if model_type:
                return (
                    f"import json as _json, os as _os, tempfile as _tmp\n"
                    f"from transformers import AutoConfig, AutoModel\n"
                    f'_raw = _json.load(open(_os.path.join("{model_path}", "config.json")))\n'
                    f'if "model_type" not in _raw:\n'
                    f'    _raw["model_type"] = "{model_type}"\n'
                    f"    _td = _tmp.mkdtemp()\n"
                    f'    _json.dump(_raw, open(_os.path.join(_td, "config.json"), "w"))\n'
                    f"    _config = AutoConfig.from_pretrained(_td, trust_remote_code=True)\n"
                    f"else:\n"
                    f'    _config = AutoConfig.from_pretrained("{model_path}", trust_remote_code=True)\n'
                    f"model = AutoModel.from_config(_config)"
                )
            else:
                return (
                    f"from transformers import AutoConfig, AutoModel\n"
                    f'_config = AutoConfig.from_pretrained("{model_path}", trust_remote_code=True)\n'
                    f"model = AutoModel.from_config(_config)"
                )

    def _generate_input_code(self, model_metadata: ModelMetadata) -> str:
        """Generate input tensor construction code based on model metadata"""
        lines = ["inputs = {"]

        for name, shape in model_metadata.input_shapes.items():
            dtype = model_metadata.input_dtypes.get(name, "int64")
            torch_dtype = self._get_torch_dtype(dtype)
            shape_tuple = f"({', '.join(map(str, shape))})"

            if dtype == "int64":
                if "input_ids" in name.lower() or "decoder_input_ids" in name.lower():
                    safe_vocab_size = self._calculate_safe_vocab_size(model_metadata)
                    value = (
                        f"torch.randint(0, {safe_vocab_size}, {shape_tuple}, "
                        f"dtype={torch_dtype}).to(device)"
                    )
                else:
                    value = f"torch.ones({shape_tuple}, dtype={torch_dtype}).to(device)"
            else:
                value = f"torch.randn({shape_tuple}, dtype={torch_dtype}).to(device)"

            lines.append(f'    "{name}": {value},')

        lines.append("}")
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
