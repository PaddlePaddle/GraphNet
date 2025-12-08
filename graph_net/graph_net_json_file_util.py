kDimensionGeneralizationPasses = "dimension_generalization_passes"
kDataTypeGeneralizationPasses = "data_type_generalization_passes"

import json
from pathlib import Path
from typing import Union


def update_json(json_path: Union[str, Path], updates: dict) -> None:
    """
    Atomically update a JSON file with the given updates.

    Args:
        json_path: Path to the JSON file
        updates: Dictionary of key-value pairs to update
    """
    json_path = Path(json_path)
    
    # Read existing JSON
    if json_path.exists():
        with open(json_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Apply updates
    metadata.update(updates)
    
    # Atomic write: write to temp file then rename
    temp_path = json_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(metadata, f, indent=4)
    temp_path.replace(json_path)
