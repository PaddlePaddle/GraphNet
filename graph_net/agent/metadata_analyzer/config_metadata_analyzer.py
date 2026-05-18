"""Config-based model analyzer implementation"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graph_net.agent.metadata_analyzer.base import BaseMetadataAnalyzer
from graph_net.agent.metadata_analyzer.model_metadata import ModelMetadata
from graph_net.agent.utils.exceptions import MetadataAnalysisError


# Cap sequence length to avoid OOM: attention is O(n²), graph extraction
# only needs a short sequence to trace the computation graph.
_MAX_SEQ_LEN = 128
# Cap image size to avoid OOM on high-resolution configs.
_MAX_IMAGE_SIZE = 512
_EMBEDDING_WEIGHT_KEYS = [
    "embeddings.word_embeddings.weight",
    "model.embed_tokens.weight",
    "roberta.embeddings.word_embeddings.weight",
    "bert.embeddings.word_embeddings.weight",
]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Unified attribute/key access for both PretrainedConfig objects and dicts."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ConfigMetadataAnalyzer(BaseMetadataAnalyzer):
    """Analyzer that extracts metadata from config.json, using transformers AutoConfig
    when available to leverage rich config object properties for architecture detection.
    """

    def analyze(self, model_dir: Path) -> ModelMetadata:
        """
        Analyze model by parsing config.json (with transformers AutoConfig if available).
        Also handles diffusers-style configs that lack a 'model_type' key but have
        '_class_name' (e.g., UNet2DConditionModel).

        Args:
            model_dir: Path to model directory

        Returns:
            ModelMetadata object

        Raises:
            MetadataAnalysisError: If analysis fails
        """
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise MetadataAnalysisError(
                f"config.json not found in {model_dir}",
                error_category="config_not_found",
            )

        try:
            # Primary path: load via AutoConfig to get a rich PretrainedConfig object
            cfg_obj = None
            try:
                from transformers import AutoConfig

                cfg_obj = AutoConfig.from_pretrained(
                    str(model_dir), trust_remote_code=True
                )
            except Exception:
                pass  # fall back to dict-only mode

            # Always parse raw dict as fallback / supplementary info
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)

            arch_type = self._classify_architecture(cfg_obj, cfg_dict)
            input_shapes, input_dtypes = self._extract_input_info(
                cfg_obj, cfg_dict, arch_type
            )

            vocab_size = (
                _cfg_get(cfg_obj, "vocab_size")
                if cfg_obj is not None
                else cfg_dict.get("vocab_size")
            )
            embedding_size = self._get_embedding_size(model_dir)
            model_id = self._get_model_id(model_dir, cfg_dict)
            model_type = (
                _cfg_get(cfg_obj, "model_type")
                if cfg_obj is not None
                else cfg_dict.get("model_type")
            )
            # If model_type is still missing, try field-based heuristic inference.
            # This handles models with incomplete config.json (e.g., prajjwal1/bert-tiny).
            if not model_type:
                model_type = self._infer_model_type_from_fields(cfg_dict)

            return ModelMetadata(
                model_id=model_id,
                input_shapes=input_shapes,
                input_dtypes=input_dtypes,
                model_type=model_type,
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                architecture_type=arch_type,
            )
        except json.JSONDecodeError as e:
            raise MetadataAnalysisError(
                f"Failed to parse config.json: {e}",
                error_category="config_parse_error",
            ) from e
        except MetadataAnalysisError:
            raise
        except Exception as e:
            raise MetadataAnalysisError(
                f"Failed to analyze model: {e}",
                error_category="metadata_analysis_failed",
            ) from e

    # ------------------------------------------------------------------
    # Architecture classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_architecture(cfg_obj: Any, cfg_dict: Dict) -> Optional[str]:
        """
        Classify model architecture type using transformers' own task mapping tables
        when available, falling back to config field inspection.

        Priority order (high → low):
          diffusion > audio > multimodal > moe > seq2seq > vision > text
        """
        model_type = (
            _cfg_get(cfg_obj, "model_type") or cfg_dict.get("model_type") or ""
        ).lower()

        # 1. Diffusion models (diffusers ecosystem)
        #    UNet2DConditionModel config has both in_channels and sample_size.
        has_in_channels = _cfg_get(cfg_obj, "in_channels") or cfg_dict.get(
            "in_channels"
        )
        has_sample_size = _cfg_get(cfg_obj, "sample_size") or cfg_dict.get(
            "sample_size"
        )
        if has_in_channels and has_sample_size:
            return "diffusion"

        # 2. Audio models
        #    Use the union of transformers' audio task mapping tables — no hardcoded list.
        try:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
                MODEL_FOR_CTC_MAPPING_NAMES,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
                MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES,
                MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES,
            )

            all_audio: set = (
                set(MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)
                | set(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES)
                | set(MODEL_FOR_CTC_MAPPING_NAMES)
                | set(MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES)
                | set(MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES)
            )
            if model_type in all_audio:
                return "audio"
        except ImportError:
            # Attribute-based fallback
            if _cfg_get(cfg_obj, "num_mel_bins") or cfg_dict.get("num_mel_bins"):
                return "audio"
            if _cfg_get(cfg_obj, "feat_extract_norm") or cfg_dict.get(
                "feat_extract_norm"
            ):
                return "audio"

        # 3. Multimodal VLMs
        #    Use transformers' multimodal task mapping tables — no hardcoded list.
        try:
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
                MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES,
            )

            all_multimodal: set = set(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES) | set(
                MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES
            )
            if model_type in all_multimodal:
                return "multimodal"
        except ImportError:
            pass
        # Fallback: check sub_configs / dict keys for vision+text pair
        if cfg_obj is not None:
            sub_configs = getattr(cfg_obj, "sub_configs", {})
            has_vision = "vision_config" in sub_configs or hasattr(
                cfg_obj, "vision_config"
            )
            has_text = "text_config" in sub_configs or hasattr(cfg_obj, "text_config")
            if has_vision and has_text:
                return "multimodal"
        elif cfg_dict.get("vision_config") and cfg_dict.get("text_config"):
            return "multimodal"

        # 4. MoE (Mixture of Experts)
        #    No common base class in transformers; detect via public config field names.
        #    We only look at non-private fields (skip leading-underscore) to avoid
        #    matching internal attributes like `_experts_implementation_internal` that
        #    exist on ALL PretrainedConfig objects.
        _MOE_PUBLIC_PREFIXES = (
            "num_experts",  # num_experts, num_experts_per_tok, num_local_experts
            "n_routed_experts",  # DeepSeek-V2/V3
            "num_local_experts",  # Mixtral (also caught by num_experts prefix)
            "num_shared_experts",
            "expert_capacity",  # switch_transformers
            "moe_topk",
            "moe_k",
            "moe_intermediate_size",
        )
        if cfg_obj is not None:
            public_fields = {k for k in vars(cfg_obj).keys() if not k.startswith("_")}
            if any(
                k.lower().startswith(p)
                for k in public_fields
                for p in _MOE_PUBLIC_PREFIXES
            ):
                return "moe"
        else:
            if any(
                k.lower().startswith(p)
                for k in cfg_dict.keys()
                for p in _MOE_PUBLIC_PREFIXES
            ):
                return "moe"

        # 5. Seq2Seq (encoder-decoder)
        #    Use the standard transformers PretrainedConfig attribute.
        #    Must be checked AFTER audio so that whisper (which also sets
        #    is_encoder_decoder=True) is classified as audio.
        is_enc_dec = (
            getattr(cfg_obj, "is_encoder_decoder", False)
            if cfg_obj is not None
            else cfg_dict.get("is_encoder_decoder", False)
        )
        if is_enc_dec:
            return "seq2seq"

        # 6. Pure vision (no vocabulary)
        has_image = (
            _cfg_get(cfg_obj, "image_size")
            or cfg_dict.get("image_size")
            or cfg_dict.get("num_channels")
        )
        has_vocab = _cfg_get(cfg_obj, "vocab_size") or cfg_dict.get("vocab_size")
        if has_image and not has_vocab:
            return "vision"

        # 7. Default: text
        return "text"

    # ------------------------------------------------------------------
    # Input shape / dtype extraction
    # ------------------------------------------------------------------

    def _extract_input_info(
        self,
        cfg_obj: Any,
        cfg_dict: Dict,
        arch_type: Optional[str],
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        """
        Build input_shapes and input_dtypes based on the detected architecture type.
        """
        if arch_type == "diffusion":
            return self._inputs_diffusion(cfg_obj, cfg_dict)
        if arch_type == "audio":
            return self._inputs_audio(cfg_obj, cfg_dict)
        if arch_type == "multimodal":
            return self._inputs_multimodal(cfg_obj, cfg_dict)
        if arch_type == "seq2seq":
            return self._inputs_seq2seq(cfg_obj, cfg_dict)
        if arch_type == "vision":
            return self._inputs_vision(cfg_obj, cfg_dict)
        # text / moe / None → standard NLP inputs
        return self._inputs_text(cfg_obj, cfg_dict)

    def _inputs_text(
        self, cfg_obj: Any, cfg_dict: Dict
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        raw_len = _cfg_get(cfg_obj, "max_position_embeddings") or cfg_dict.get(
            "max_position_embeddings", 512
        )
        seq_len = min(int(raw_len), _MAX_SEQ_LEN)
        shapes = {
            "input_ids": [1, seq_len],
            "attention_mask": [1, seq_len],
        }
        dtypes = {
            "input_ids": "int64",
            "attention_mask": "int64",
        }
        return shapes, dtypes

    def _inputs_seq2seq(
        self, cfg_obj: Any, cfg_dict: Dict
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        enc_len = 64
        dec_len = 32
        shapes = {
            "input_ids": [1, enc_len],
            "attention_mask": [1, enc_len],
            "decoder_input_ids": [1, dec_len],
        }
        dtypes = {
            "input_ids": "int64",
            "attention_mask": "int64",
            "decoder_input_ids": "int64",
        }
        return shapes, dtypes

    def _inputs_vision(
        self, cfg_obj: Any, cfg_dict: Dict
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        raw_size = _cfg_get(cfg_obj, "image_size") or cfg_dict.get("image_size", 224)
        if isinstance(raw_size, (list, tuple)):
            raw_size = raw_size[0]
        image_size = min(int(raw_size), _MAX_IMAGE_SIZE)
        num_channels = _cfg_get(cfg_obj, "num_channels") or cfg_dict.get(
            "num_channels", 3
        )
        shapes = {"pixel_values": [1, int(num_channels), image_size, image_size]}
        dtypes = {"pixel_values": "float32"}
        return shapes, dtypes

    def _inputs_multimodal(
        self, cfg_obj: Any, cfg_dict: Dict
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        # Text branch — prefer text_config.max_position_embeddings (e.g., CLIP has 77)
        txt_cfg = _cfg_get(cfg_obj, "text_config") or cfg_dict.get("text_config", {})
        if hasattr(txt_cfg, "max_position_embeddings"):
            raw_len = txt_cfg.max_position_embeddings
        elif isinstance(txt_cfg, dict):
            raw_len = txt_cfg.get("max_position_embeddings", None)
        else:
            raw_len = None
        if raw_len is None:
            raw_len = _cfg_get(cfg_obj, "max_position_embeddings") or cfg_dict.get(
                "max_position_embeddings", 512
            )
        seq_len = min(int(raw_len), _MAX_SEQ_LEN)

        # Vision branch — prefer sub vision_config
        vis_cfg = _cfg_get(cfg_obj, "vision_config") or cfg_dict.get(
            "vision_config", {}
        )
        if hasattr(vis_cfg, "image_size"):
            raw_size = vis_cfg.image_size
            num_channels = getattr(vis_cfg, "num_channels", 3)
        elif isinstance(vis_cfg, dict):
            raw_size = vis_cfg.get("image_size", 224)
            num_channels = vis_cfg.get("num_channels", 3)
        else:
            raw_size = _cfg_get(cfg_obj, "image_size") or cfg_dict.get(
                "image_size", 224
            )
            num_channels = _cfg_get(cfg_obj, "num_channels") or cfg_dict.get(
                "num_channels", 3
            )
        if isinstance(raw_size, (list, tuple)):
            raw_size = raw_size[0]
        image_size = min(int(raw_size), _MAX_IMAGE_SIZE)

        shapes = {
            "input_ids": [1, seq_len],
            "attention_mask": [1, seq_len],
            "pixel_values": [1, int(num_channels), image_size, image_size],
        }
        dtypes = {
            "input_ids": "int64",
            "attention_mask": "int64",
            "pixel_values": "float32",
        }
        return shapes, dtypes

    def _inputs_audio(
        self, cfg_obj: Any, cfg_dict: Dict
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        model_type = (
            _cfg_get(cfg_obj, "model_type") or cfg_dict.get("model_type") or ""
        ).lower()

        if model_type == "whisper":
            num_mel_bins = int(
                _cfg_get(cfg_obj, "num_mel_bins") or cfg_dict.get("num_mel_bins", 80)
            )
            # chunk_length (seconds) × sample_rate / hop_length ≈ frames
            # whisper default: 30s × 16000 / 160 = 3000 frames
            frames = 3000
            shapes = {
                "input_features": [1, num_mel_bins, frames],
                # Whisper is encoder-decoder; explicitly pass decoder_input_ids
                # to avoid internal conflict between decoder_input_ids and
                # decoder_inputs_embeds auto-generation.
                "decoder_input_ids": [1, 1],
            }
            dtypes = {
                "input_features": "float32",
                "decoder_input_ids": "int64",
            }
        elif model_type == "clap":
            shapes = {"input_features": [1, 1, 1001, 64]}
            dtypes = {"input_features": "float32"}
        else:
            # wav2vec2, hubert, unispeech, wavlm, sew, etc. — 1 second at 16 kHz
            shapes = {"input_values": [1, 16000]}
            dtypes = {"input_values": "float32"}
        return shapes, dtypes

    def _inputs_diffusion(
        self, cfg_obj: Any, cfg_dict: Dict
    ) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
        in_channels = int(
            _cfg_get(cfg_obj, "in_channels") or cfg_dict.get("in_channels", 4)
        )
        sample_size_raw = _cfg_get(cfg_obj, "sample_size") or cfg_dict.get(
            "sample_size", 64
        )
        if isinstance(sample_size_raw, (list, tuple)):
            sample_size = int(sample_size_raw[0])
        else:
            sample_size = int(sample_size_raw)
        cross_dim = int(
            _cfg_get(cfg_obj, "cross_attention_dim")
            or cfg_dict.get("cross_attention_dim", 768)
        )
        shapes = {
            "sample": [1, in_channels, sample_size, sample_size],
            "timestep": [1],
            "encoder_hidden_states": [1, 77, cross_dim],
        }
        dtypes = {
            "sample": "float32",
            "timestep": "int64",
            "encoder_hidden_states": "float32",
        }
        return shapes, dtypes

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _infer_model_type(self, config: Dict) -> Optional[str]:
        """Infer model type from raw config dict (legacy path)."""
        if "model_type" in config:
            return config["model_type"]
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

    @staticmethod
    def _infer_model_type_from_fields(cfg_dict: Dict) -> Optional[str]:
        """
        Last-resort model_type inference based on field name signatures.
        Used when config.json has neither 'model_type' nor 'architectures'.
        """
        keys = set(cfg_dict.keys())
        # BERT: type_vocab_size is unique to BERT/RoBERTa family
        if "type_vocab_size" in keys:
            return "bert"
        # GPT-2: n_head / n_layer naming
        if "n_head" in keys and "n_layer" in keys:
            return "gpt2"
        # Generic transformer-like text model
        if {
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
        } <= keys:
            return "bert"
        return None

    def _get_model_id(self, model_dir: Path, config: Dict) -> str:
        """Get model ID from directory or config."""
        if "name_or_path" in config:
            return config["name_or_path"]
        return model_dir.name

    def _get_embedding_size(self, model_dir: Path) -> Optional[int]:
        """Get actual embedding layer size from model weights."""
        model_file = self._find_model_file(model_dir)
        if not model_file:
            return None
        if model_file.suffix == ".safetensors":
            return self._get_embedding_size_from_safetensors(model_file)
        else:
            return self._get_embedding_size_from_pytorch(model_file)

    def _find_model_file(self, model_dir: Path) -> Optional[Path]:
        """Find model weight file (pytorch_model*.bin or model.safetensors)."""
        model_files = list(model_dir.glob("pytorch_model*.bin"))
        if not model_files:
            model_files = list(model_dir.glob("model.safetensors"))
        return model_files[0] if model_files else None

    def _get_embedding_size_from_safetensors(self, model_file: Path) -> Optional[int]:
        """Extract embedding size from safetensors file."""
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
        """Extract embedding size from PyTorch .bin file."""
        try:
            import torch

            state_dict = torch.load(model_file, map_location="cpu")

            for key in _EMBEDDING_WEIGHT_KEYS:
                if key in state_dict:
                    tensor = state_dict[key]
                    if tensor is not None and len(tensor.shape) >= 1:
                        return int(tensor.shape[0])

            for key, tensor in state_dict.items():
                if "embedding" in key.lower() and "weight" in key.lower():
                    if tensor is not None and len(tensor.shape) >= 1:
                        return int(tensor.shape[0])
        except Exception:
            pass
        return None
