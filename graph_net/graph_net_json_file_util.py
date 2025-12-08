from pathlib import Path
import json

kDimensionGeneralizationPasses = "dimension_generalization_passes"
kSymbolicDimensionReifier = "symbolic_dimension_reifier"


def read_json(model_path):
    graph_net_json_file_path = Path(f"{model_path}/graph_net.json")
    return json.loads(graph_net_json_file_path.read_text())


def update_json(model_path, field, value):
    graph_net_json_file_path = Path(f"{model_path}/graph_net.json")
    graph_net_json = json.loads(graph_net_json_file_path.read_text())
    graph_net_json[field] = value
    graph_net_json_file_path.write_text(json.dumps(graph_net_json, indent=4))
