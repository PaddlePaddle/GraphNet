import os
import json
import logging
from typing import Dict, Any

class ConfigAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("ConfigAnalyzer")

    def analyze(self, model_dir: str) -> Dict[str, Any]:
        """
        Analyze config.json to infer input specifications.
        """
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        architecture = config.get("architectures", ["Unknown"])[0]
        self.logger.info(f"Detected architecture: {architecture}")

        # Heuristic rules
        meta_info = {
            "architecture": architecture,
            "input_shape": [1, 128], # Default batch size 1, seq len 128
            "input_dtype": "int64",
            "task_type": "nlp"
        }

        # Refine based on architecture
        if "Bert" in architecture or "Roberta" in architecture:
            meta_info["input_names"] = ["input_ids", "attention_mask", "token_type_ids"]
        elif "Gpt" in architecture or "Llama" in architecture:
            meta_info["input_names"] = ["input_ids", "attention_mask"]
        elif "ResNet" in architecture or "ViT" in architecture:
            meta_info["task_type"] = "cv"
            meta_info["input_shape"] = [1, 3, 224, 224]
            meta_info["input_dtype"] = "float32"
            meta_info["input_names"] = ["pixel_values"]

        return meta_info
