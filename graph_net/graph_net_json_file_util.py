import json
from pathlib import Path
from typing import Union

kDimensionGeneralizationPasses = "dimension_generalization_passes"
kDataTypeGeneralizationPasses = "data_type_generalization_passes"
kSymbolicDimensionReifier = "symbolic_dimension_reifier"


def read_json(model_path):
    """
    Read JSON from graph_net.json file.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary containing JSON data
    """
    graph_net_json_file_path = Path(f"{model_path}/graph_net.json")
    return json.loads(graph_net_json_file_path.read_text())


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


# Backward compatibility: old interface using model_path, field, value
def update_json_legacy(model_path, field, value):
    """
    Legacy interface for updating a single field in graph_net.json.

    Args:
        model_path: Path to model directory
        field: Field name to update
        value: Value to set
    """
    graph_net_json_file_path = Path(f"{model_path}/graph_net.json")
    graph_net_json = json.loads(graph_net_json_file_path.read_text())
    graph_net_json[field] = value
    graph_net_json_file_path.write_text(json.dumps(graph_net_json, indent=4))
