"""Template-based code generator implementation"""

import json
from pathlib import Path
from typing import Optional

from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata
from graph_net.agent.code_generator.base import BaseCodeGenerator
from graph_net.agent.model_type_utils import (
    VLM_FAMILY_GEMMA3,
    VLM_FAMILY_INTERNVL,
    VLM_FAMILY_LLAVA,
    VLM_FAMILY_QWEN,
    get_vlm_family,
    select_auto_model_class,
)
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

        if arch == "diffusion":
            return (
                f"from diffusers import UNet2DConditionModel\n"
                f'_config = UNet2DConditionModel.load_config("{model_path}")\n'
                f"model = UNet2DConditionModel.from_config(_config)"
            )

        model_class = select_auto_model_class(
            model_metadata.model_type, model_metadata.architecture_type
        )
        model_type = model_metadata.model_type
        if model_type and self._config_missing_model_type(model_dir):
            return (
                f"import json as _json, os as _os, tempfile as _tmp\n"
                f"from transformers import AutoConfig, {model_class}\n"
                f'_raw = _json.load(open(_os.path.join("{model_path}", "config.json")))\n'
                f'_raw["model_type"] = "{model_type}"\n'
                f"_td = _tmp.mkdtemp()\n"
                f'_json.dump(_raw, open(_os.path.join(_td, "config.json"), "w"))\n'
                f"_config = AutoConfig.from_pretrained(_td, trust_remote_code=True)\n"
                f"model = {model_class}.from_config(_config)"
            )

        return (
            f"from transformers import AutoConfig, {model_class}\n"
            f'_config = AutoConfig.from_pretrained("{model_path}", trust_remote_code=True)\n'
            f"model = {model_class}.from_config(_config)"
        )

    @staticmethod
    def _config_missing_model_type(model_dir: Path) -> bool:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return False
        try:
            return "model_type" not in json.loads(
                config_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False

    def _generate_input_code(self, model_metadata: ModelMetadata) -> str:
        """Generate input tensor construction code based on model metadata"""
        family = get_vlm_family(model_metadata.model_type)
        generators = {
            VLM_FAMILY_QWEN: self._generate_qwen_vlm_input_code,
            VLM_FAMILY_LLAVA: self._generate_llava_input_code,
            VLM_FAMILY_GEMMA3: self._generate_gemma3_input_code,
            VLM_FAMILY_INTERNVL: self._generate_internvl_input_code,
        }
        generator = generators.get(family, self._generate_generic_input_code)
        return generator(model_metadata)

    def _generate_generic_input_code(self, model_metadata: ModelMetadata) -> str:
        """Generate generic input tensor construction code."""
        lines = ["inputs = {"]

        for name, shape in model_metadata.input_shapes.items():
            dtype = model_metadata.input_dtypes.get(name, "int64")
            value = self._generate_tensor_value(name, shape, dtype, model_metadata)
            lines.append(f'    "{name}": {value},')

        lines.append("}")
        return "\n".join(lines)

    def _generate_tensor_value(
        self,
        name: str,
        shape: list,
        dtype: str,
        model_metadata: ModelMetadata,
    ) -> str:
        torch_dtype = self._get_torch_dtype(dtype)
        shape_tuple = self._shape_tuple(shape)

        if dtype == "int64":
            if "input_ids" in name.lower() or "decoder_input_ids" in name.lower():
                safe_vocab_size = self._calculate_safe_vocab_size(model_metadata)
                return (
                    f"torch.randint(0, {safe_vocab_size}, {shape_tuple}, "
                    f"dtype={torch_dtype}).to(device)"
                )
            return f"torch.ones({shape_tuple}, dtype={torch_dtype}).to(device)"
        return f"torch.randn({shape_tuple}, dtype={torch_dtype}).to(device)"

    def _generate_qwen_vlm_input_code(self, model_metadata: ModelMetadata) -> str:
        lines = []
        for name, shape in model_metadata.input_shapes.items():
            dtype = model_metadata.input_dtypes.get(name, "int64")
            if name == "image_grid_thw":
                lines.append(
                    "image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long).to(device)"
                )
            elif name == "pixel_values":
                lines.append(
                    f"pixel_values = torch.randn({self._shape_tuple(shape)}, dtype=torch.float32).to(device)"
                )
            elif name == "input_ids":
                safe_vocab_size = self._calculate_safe_vocab_size(model_metadata)
                lines.append(
                    f"input_ids = torch.randint(0, {safe_vocab_size}, {self._shape_tuple(shape)}, dtype=torch.int64).to(device)"
                )
            elif name == "attention_mask":
                lines.append(
                    f"attention_mask = torch.ones({self._shape_tuple(shape)}, dtype=torch.int64).to(device)"
                )
            else:
                lines.append(
                    f"{name} = {self._generate_tensor_value(name, shape, dtype, model_metadata)}"
                )
        lines.extend(self._input_dict_lines(model_metadata))
        return "\n".join(lines)

    def _generate_llava_input_code(self, model_metadata: ModelMetadata) -> str:
        lines = self._generate_vlm_base_assignment_lines(model_metadata)
        seq_len = self._input_seq_len(model_metadata)
        image_shape = model_metadata.input_shapes.get("pixel_values", [1, 3, 224, 224])
        image_size = image_shape[-1]
        lines.extend(
            [
                'image_token_index = int(getattr(_config, "image_token_index", 32000))',
                'patch_size = int(getattr(getattr(_config, "vision_config", _config), "patch_size", 14))',
                f"image_size = {image_size}",
                "num_image_tokens = (image_size // patch_size) ** 2",
                'if getattr(_config, "vision_feature_select_strategy", None) == "full":',
                "    num_image_tokens += 1",
                f"num_image_tokens = min(num_image_tokens, {seq_len})",
                "input_ids[:, :num_image_tokens] = image_token_index",
            ]
        )
        lines.extend(self._input_dict_lines(model_metadata))
        return "\n".join(lines)

    def _generate_gemma3_input_code(self, model_metadata: ModelMetadata) -> str:
        lines = self._generate_vlm_base_assignment_lines(model_metadata)
        seq_len = self._input_seq_len(model_metadata)
        lines.extend(
            [
                'image_token_index = int(getattr(_config, "image_token_index", 262144))',
                'num_image_tokens = int(getattr(_config, "mm_tokens_per_image", 256))',
                f"num_image_tokens = min(num_image_tokens, {seq_len})",
                "input_ids[:, :num_image_tokens] = image_token_index",
            ]
        )
        lines.extend(self._input_dict_lines(model_metadata))
        return "\n".join(lines)

    def _generate_internvl_input_code(self, model_metadata: ModelMetadata) -> str:
        lines = self._generate_vlm_base_assignment_lines(model_metadata)
        seq_len = self._input_seq_len(model_metadata)
        if "image_flags" in model_metadata.input_shapes:
            lines.append(
                "image_flags = torch.ones((1, 1), dtype=torch.long).to(device)"
            )
        lines.extend(
            [
                'num_image_token = int(getattr(_config, "num_image_token", 256))',
                'image_token_id = int(getattr(_config, "img_context_token_id", getattr(_config, "image_token_index", 0)))',
                f"num_image_token = min(num_image_token, {seq_len})",
                "input_ids[:, :num_image_token] = image_token_id",
            ]
        )
        lines.extend(self._input_dict_lines(model_metadata))
        return "\n".join(lines)

    def _generate_vlm_base_assignment_lines(
        self, model_metadata: ModelMetadata
    ) -> list:
        lines = []
        for name, shape in model_metadata.input_shapes.items():
            dtype = model_metadata.input_dtypes.get(name, "int64")
            if name == "image_flags":
                continue
            lines.append(
                f"{name} = {self._generate_tensor_value(name, shape, dtype, model_metadata)}"
            )
        return lines

    @staticmethod
    def _shape_tuple(shape: list) -> str:
        return f"({', '.join(map(str, shape))})"

    @staticmethod
    def _input_seq_len(model_metadata: ModelMetadata) -> int:
        return int(model_metadata.input_shapes.get("input_ids", [1, 0])[1])

    @staticmethod
    def _input_dict_lines(model_metadata: ModelMetadata) -> list:
        lines = ["inputs = {"]
        for name in model_metadata.input_shapes:
            lines.append(f'    "{name}": {name},')
        lines.append("}")
        return lines

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
