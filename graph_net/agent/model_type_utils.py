"""Shared model_type helpers for agent metadata analysis and code generation."""

from typing import Optional, Tuple

VLM_FAMILY_QWEN = "qwen_vlm"
VLM_FAMILY_LLAVA = "llava"
VLM_FAMILY_GEMMA3 = "gemma3"
VLM_FAMILY_INTERNVL = "internvl"

VLM_MODEL_TYPE_TO_FAMILY = {
    "qwen2_vl": VLM_FAMILY_QWEN,
    "qwen2_5_vl": VLM_FAMILY_QWEN,
    "qwen3_vl": VLM_FAMILY_QWEN,
    "llava": VLM_FAMILY_LLAVA,
    "gemma3": VLM_FAMILY_GEMMA3,
    "internvl": VLM_FAMILY_INTERNVL,
    "internvl_chat": VLM_FAMILY_INTERNVL,
}
IMAGE_TOKEN_VLM_MODEL_TYPES = frozenset(VLM_MODEL_TYPE_TO_FAMILY)

AUDIO_AUTO_MAPPING_NAMES = (
    "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES",
    "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
    "MODEL_FOR_CTC_MAPPING_NAMES",
    "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES",
    "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES",
)
IMAGE_TEXT_TO_TEXT_AUTO_MAPPING_NAMES = ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",)
MULTIMODAL_LM_AUTO_MAPPING_NAMES = ("MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES",)
OBJECT_DETECTION_AUTO_MAPPING_NAMES = ("MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES",)
CAUSAL_LM_AUTO_MAPPING_NAMES = ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",)
MASKED_LM_AUTO_MAPPING_NAMES = ("MODEL_FOR_MASKED_LM_MAPPING_NAMES",)


def normalize_model_type(model_type: Optional[str]) -> str:
    return (model_type or "").lower()


def get_vlm_family(model_type: Optional[str]) -> Optional[str]:
    return VLM_MODEL_TYPE_TO_FAMILY.get(normalize_model_type(model_type))


def is_image_token_vlm(model_type: Optional[str]) -> bool:
    return get_vlm_family(model_type) is not None


def model_type_in_auto_mapping(
    model_type: Optional[str], mapping_names: Tuple[str, ...]
) -> bool:
    normalized = normalize_model_type(model_type)
    if not normalized:
        return False
    try:
        from transformers.models.auto import modeling_auto
    except ImportError:
        return False
    return any(
        normalized in getattr(modeling_auto, mapping_name, {})
        for mapping_name in mapping_names
    )


def is_audio_model_type(model_type: Optional[str]) -> bool:
    return model_type_in_auto_mapping(model_type, AUDIO_AUTO_MAPPING_NAMES)


def is_multimodal_model_type(model_type: Optional[str]) -> bool:
    return is_image_token_vlm(model_type) or model_type_in_auto_mapping(
        model_type,
        IMAGE_TEXT_TO_TEXT_AUTO_MAPPING_NAMES + MULTIMODAL_LM_AUTO_MAPPING_NAMES,
    )


def is_object_detection_model_type(model_type: Optional[str]) -> bool:
    return model_type_in_auto_mapping(model_type, OBJECT_DETECTION_AUTO_MAPPING_NAMES)


def is_causal_lm_model_type(model_type: Optional[str]) -> bool:
    return model_type_in_auto_mapping(
        model_type, CAUSAL_LM_AUTO_MAPPING_NAMES
    ) and not model_type_in_auto_mapping(model_type, MASKED_LM_AUTO_MAPPING_NAMES)


def select_auto_model_class(
    model_type: Optional[str], architecture_type: Optional[str]
) -> str:
    normalized_arch = architecture_type or ""
    if normalized_arch == "seq2seq":
        return "AutoModelForSeq2SeqLM"
    if get_vlm_family(model_type) or model_type_in_auto_mapping(
        model_type, IMAGE_TEXT_TO_TEXT_AUTO_MAPPING_NAMES
    ):
        return "AutoModelForImageTextToText"
    if model_type_in_auto_mapping(model_type, MULTIMODAL_LM_AUTO_MAPPING_NAMES):
        return "AutoModelForMultimodalLM"
    if is_object_detection_model_type(model_type):
        return "AutoModelForObjectDetection"
    if normalized_arch in {"text", "moe"} and is_causal_lm_model_type(model_type):
        return "AutoModelForCausalLM"
    return "AutoModel"
