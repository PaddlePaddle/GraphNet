"""Shared exact VLM model_type family mapping for agent input construction."""

from typing import Optional

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


def get_vlm_family(model_type: Optional[str]) -> Optional[str]:
    return VLM_MODEL_TYPE_TO_FAMILY.get((model_type or "").lower())


def is_image_token_vlm(model_type: Optional[str]) -> bool:
    return get_vlm_family(model_type) is not None
