import argparse
import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.export import export
import graph_net.torch.extractor.utils as utils
from graph_net.torch.extractor.utils import apply_templates
from transformers import AutoTokenizer, AutoModelForCausalLM


def main(model_name: str, model_path: str, text: str) -> None:
    # Load tokenizer and model from HuggingFace hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.eval()
    for name, module in model.named_modules():
        print(name, "→", module.__class__.__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare example input
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Export model
    try:
        exported = export(model, kwargs=inputs, args=())
    except Exception as e:
        print(f"Error exporting model {model_name}: {e}")
        return

    params = exported.state_dict

    # Generate and save model code
    base_code = exported.graph_module.__str__()
    write_code = apply_templates(base_code)

    os.makedirs(model_path, exist_ok=True)

    with open(f"{model_path}/model.py", "w") as fp:
        fp.write(write_code)

    # Save metadata
    metadata = {
        "framework": "torch",
        "num_devices_required": 1,
        "num_nodes_required": 1
    }

    with open(f"{model_path}/attribute.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Save tensor metadata and constraints
    converted = utils.convert_state_and_inputs(params, exported.example_inputs)
    utils.save_converted_to_text(
        converted,
        file_path=f"{model_path}/source_tensor_meta.py"
    )
    utils.save_constraints_text(
        converted,
        file_path=f"{model_path}/input_tensor_constraints.py"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export HuggingFace transformer models to txt"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name from HuggingFace hub"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Directory to save the exported model"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello world",
        help="Sample input text"
    )
    args = parser.parse_args()
    main(model_name=args.model_name, model_path=args.model_path, text=args.text)