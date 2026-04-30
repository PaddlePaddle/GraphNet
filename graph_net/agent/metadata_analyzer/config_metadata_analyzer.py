"""Config-based model analyzer implementation"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from graph_net.agent.metadata_analyzer.base import BaseMetadataAnalyzer
from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata
from graph_net.agent.utils.exceptions import AnalysisError


# Common embedding weight keys in different model architectures
_EMBEDDING_WEIGHT_KEYS = [
    "embeddings.word_embeddings.weight",
    "model.embed_tokens.weight",
    "roberta.embeddings.word_embeddings.weight",
    "bert.embeddings.word_embeddings.weight",
]


class ConfigMetadataAnalyzer(BaseMetadataAnalyzer):
    """Analyzer that extracts metadata from config.json"""

    def analyze(self, model_dir: Path, max_param_b: float = 20.0) -> ModelMetadata:
        """
        Analyze model by parsing config.json

        Args:
            model_dir:   Path to model directory
            max_param_b: Maximum allowed estimated parameter count in billions.
                         Models exceeding this limit are rejected with AnalysisError.
                         Default 20B; set based on (total_RAM × 0.7 / workers / 4).

        Returns:
            ModelMetadata object

        Raises:
            AnalysisError: If analysis fails or model is too large
        """
        config_path = model_dir / "config.json"

        if not config_path.exists():
            raise AnalysisError(f"config.json not found in {model_dir}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Extract model type
            model_type = self._infer_model_type(config)

            # Reject models that are too large to load even with random weights
            param_b = self._estimate_param_count_billion(config)
            if param_b > max_param_b:
                raise AnalysisError(
                    f"Model too large to extract: estimated {param_b:.1f}B parameters "
                    f"(limit {max_param_b:.1f}B). "
                    f"Loading random fp32 weights would require ~{param_b * 4:.0f}GB RAM."
                )

            # Extract input shapes and dtypes
            input_shapes, input_dtypes = self._extract_input_info(config)

            # Extract vocab_size
            vocab_size = config.get("vocab_size")

            # Try to get actual embedding size from model weights
            embedding_size = self._get_embedding_size(model_dir)

            # Get model_id from directory name or config
            model_id = self._get_model_id(model_dir, config)

            return ModelMetadata(
                model_id=model_id,
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                model_type=model_type,
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                oom_risk=self._estimate_oom_risk(config),
            )
        except json.JSONDecodeError as e:
            raise AnalysisError(f"Failed to parse config.json: {e}") from e
        except Exception as e:
            raise AnalysisError(f"Failed to analyze model: {e}") from e

    def _infer_model_type(self, config: Dict) -> Optional[str]:
        """Infer model type from config"""
        # Check common model type indicators
        if "model_type" in config:
            return config["model_type"]

        # Check architecture field
        if "architectures" in config and config["architectures"]:
            arch = config["architectures"][0].lower()
            if "bert" in arch:
                return "bert"
            elif "gpt" in arch or "llama" in arch:
                return "gpt"
            elif "resnet" in arch:
                return "resnet"
            elif "vit" in arch or "vision" in arch:
                return "vit"

        return None

    def _extract_input_info(
        self, config: Dict
    ) -> tuple[Dict[str, List[int]], Dict[str, str]]:
        """
        Extract input shapes and dtypes from config

        Returns:
            Tuple of (input_shapes, input_dtypes)
        """
        input_shapes = {}
        input_dtypes = {}

        # Common patterns for NLP models
        MAX_SEQ_LEN = 2048  # Cap sequence length to avoid OOM on large-context models
        if "max_position_embeddings" in config or "vocab_size" in config:
            # NLP model (BERT, GPT, etc.)
            max_length = min(config.get("max_position_embeddings", 512), MAX_SEQ_LEN)
            batch_size = 1
            input_shapes["input_ids"] = [batch_size, max_length]
            input_dtypes["input_ids"] = "int64"

            # Add attention_mask if present
            if "attention_mask" not in input_shapes:
                input_shapes["attention_mask"] = [batch_size, max_length]
                input_dtypes["attention_mask"] = "int64"

        # Common patterns for vision models
        elif "image_size" in config or "num_channels" in config:
            # Vision model (ResNet, ViT, etc.)
            image_size = config.get("image_size", 224)
            num_channels = config.get("num_channels", 3)
            batch_size = 1
            input_shapes["pixel_values"] = [
                batch_size,
                num_channels,
                image_size,
                image_size,
            ]
            input_dtypes["pixel_values"] = "float32"

        # Fallback: use default values
        if not input_shapes:
            # Default to common NLP input
            input_shapes["input_ids"] = [1, 128]
            input_dtypes["input_ids"] = "int64"

        return input_shapes, input_dtypes

    @staticmethod
    def _estimate_param_count_billion(config: Dict) -> float:
        """Rough estimate of model parameter count in billions.

        Formula covers standard Transformer decoder/encoder:
          - Per layer: attention (4 × hidden²) + FFN (3 × hidden × intermediate, SwiGLU style)
          - Embedding: 2 × vocab_size × hidden_size (input + output, unshared)
        MoE models: total params ≈ num_experts × expert_params, but only a few
        experts are active per token.  We use total params here because all expert
        weights must be loaded into memory even when inactive.
        """
        hidden_size = config.get("hidden_size", 768) or 768
        num_layers = config.get("num_hidden_layers", 12) or 12
        intermediate_size = (
            config.get("intermediate_size") or config.get("ffn_dim") or hidden_size * 4
        )
        vocab_size = config.get("vocab_size", 32000) or 32000

        # MoE: total expert count (all experts loaded into RAM)
        num_experts = (
            config.get("num_experts")
            or config.get("num_local_experts")
            or config.get("moe_num_experts")
            or 1
        )

        attn_params = 4 * hidden_size * hidden_size          # Q, K, V, O
        ffn_params = 3 * hidden_size * intermediate_size     # gate, up, down (SwiGLU)
        expert_ffn_params = ffn_params * int(num_experts)
        embed_params = 2 * vocab_size * hidden_size          # input + output embedding

        total = num_layers * (attn_params + expert_ffn_params) + embed_params
        return total / 1e9

    @staticmethod
    def _estimate_oom_risk(config: Dict) -> str:
        """Estimate GPU OOM risk from config fields.

        Returns 'low', 'medium', or 'high'.
        'high' means the model should fall back to CPU to avoid OOM.
        """
        # Large models (>7B params) need >28GB GPU RAM in fp32 → CPU fallback
        param_b = ConfigMetadataAnalyzer._estimate_param_count_billion(config)
        if param_b > 7.0:
            return "high"
        if param_b > 3.0:
            return "medium"

        vocab_size = config.get("vocab_size", 0) or 0
        hidden_size = config.get("hidden_size", 768) or 768
        num_layers = config.get("num_hidden_layers", 12) or 12
        num_experts = (
            config.get("num_experts")
            or config.get("num_local_experts")
            or 1
        )
        # Raw context window: models may allocate internal attention buffers
        # based on max_position_embeddings regardless of actual input length
        raw_ctx_len = config.get("max_position_embeddings", 512) or 512
        seq_len = min(raw_ctx_len, 2048)

        # Models with very large context may allocate causal mask buffers of
        # size max_position_embeddings × max_position_embeddings internally
        if raw_ctx_len > 65536:
            return "high"
        if raw_ctx_len > 16384:
            return "medium"

        # lm_head output tensor (MB): batch=1 × seq_len × vocab_size × fp32
        lm_head_mb = seq_len * vocab_size * 4 / 1024**2
        # Activations estimate (MB): layers × seq_len × hidden_size × fp32
        activation_mb = num_layers * seq_len * hidden_size * 4 / 1024**2
        # MoE amplification (cap at 8 concurrent experts)
        moe_factor = min(int(num_experts), 8) if int(num_experts) > 1 else 1

        total_est_mb = (lm_head_mb + activation_mb) * moe_factor

        if total_est_mb > 8000:
            return "high"
        if total_est_mb > 3000:
            return "medium"
        return "low"

    def _get_model_id(self, model_dir: Path, config: Dict) -> str:
        """Get model ID from directory or config"""
        # Try to get from config first
        if "name_or_path" in config:
            return config["name_or_path"]

        # Fallback to directory name
        return model_dir.name

    def _get_embedding_size(self, model_dir: Path) -> Optional[int]:
        """Get actual embedding layer size from model weights"""
        model_file = self._find_model_file(model_dir)
        if not model_file:
            return None

        if model_file.suffix == ".safetensors":
            return self._get_embedding_size_from_safetensors(model_file)
        else:
            return self._get_embedding_size_from_pytorch(model_file)

    def _find_model_file(self, model_dir: Path) -> Optional[Path]:
        """Find model weight file (pytorch_model*.bin or model.safetensors)"""
        model_files = list(model_dir.glob("pytorch_model*.bin"))
        if not model_files:
            model_files = list(model_dir.glob("model.safetensors"))
        return model_files[0] if model_files else None

    def _get_embedding_size_from_safetensors(self, model_file: Path) -> Optional[int]:
        """Extract embedding size from safetensors file"""
        try:
            from safetensors import safe_open

            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in _EMBEDDING_WEIGHT_KEYS:
                    if key in f.keys():
                        tensor = f.get_tensor(key)
                        if tensor is not None and len(tensor.shape) >= 1:
                            return int(tensor.shape[0])
        except Exception:
            pass
        return None

    def _get_embedding_size_from_pytorch(self, model_file: Path) -> Optional[int]:
        """Extract embedding size from PyTorch .bin file"""
        try:
            import torch

            state_dict = torch.load(model_file, map_location="cpu")

            # Check known embedding keys first
            for key in _EMBEDDING_WEIGHT_KEYS:
                if key in state_dict:
                    tensor = state_dict[key]
                    if tensor is not None and len(tensor.shape) >= 1:
                        return int(tensor.shape[0])

            # Fallback: search by pattern
            for key, tensor in state_dict.items():
                if "embedding" in key.lower() and "weight" in key.lower():
                    if tensor is not None and len(tensor.shape) >= 1:
                        return int(tensor.shape[0])
        except Exception:
            pass
        return None
