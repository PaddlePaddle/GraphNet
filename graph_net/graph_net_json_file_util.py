import json
from pathlib import Path

kDimensionGeneralizationPasses = "dimension_generalization_passes"
kDataTypeGeneralizationPasses = "data_type_generalization_passes"
kSymbolicDimensionReifier = "symbolic_dimension_reifier"

# Fields for dtype generalization metadata
kDtypeGeneralizationTargetDtype = "dtype_generalization_target_dtype"
kDtypeGeneralizationPrecision = "dtype_generalization_precision"
kDtypeGeneralizationGenerated = "dtype_generalization_generated"


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


def update_json(model_path, field, value):
    """
    Update a single field in graph_net.json.

    Args:
        model_path: Path to model directory or graph_net.json file
        field: Field name to update
        value: Value to set
    """
    if isinstance(model_path, (str, Path)):
        model_path = Path(model_path)
        # If it's a file path, use it directly; otherwise assume it's a directory
        if model_path.suffix == ".json":
            graph_net_json_file_path = model_path
        else:
            graph_net_json_file_path = model_path / "graph_net.json"
    else:
        graph_net_json_file_path = Path(f"{model_path}/graph_net.json")

    # Read existing JSON
    if graph_net_json_file_path.exists():
        with open(graph_net_json_file_path, "r") as f:
            graph_net_json = json.load(f)
    else:
        graph_net_json = {}

    # Update field
    graph_net_json[field] = value

    # Atomic write: write to temp file then rename
    temp_path = graph_net_json_file_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(graph_net_json, f, indent=4)
    temp_path.replace(graph_net_json_file_path)
