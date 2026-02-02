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

    def analyze(self, model_dir: Path) -> ModelMetadata:
        """
        Analyze model by parsing config.json

        Args:
            model_dir: Path to model directory

        Returns:
            ModelMetadata object

        Raises:
            AnalysisError: If analysis fails
        """
        config_path = model_dir / "config.json"

        if not config_path.exists():
            raise AnalysisError(f"config.json not found in {model_dir}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Extract model type
            model_type = self._infer_model_type(config)

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
        if "max_position_embeddings" in config or "vocab_size" in config:
            # NLP model (BERT, GPT, etc.)
            max_length = config.get("max_position_embeddings", 512)
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
