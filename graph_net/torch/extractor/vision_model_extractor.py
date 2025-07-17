import argparse
import os
import json
import torch
import torchvision
from torchvision import transforms
from torch.export import export
from torch import nn
import graph_net.torch.extractor.utils as utils
from graph_net.torch.extractor.utils import convert_param_name, indent_with_tab, apply_templates


def main(key, model_path):
    # Normalization parameters for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Create dummy input
    batch_size = 1
    height, width = 224, 224  # Standard ImageNet size
    num_channels = 3
    random_input = torch.rand(batch_size, num_channels, height, width)
    normalized_input = normalize(random_input)

    # Get and initialize model
    try:
        model = torchvision.models.get_model(key, weights="DEFAULT")
    except ValueError as e:
        print(f"Error loading model {key}: {e}")
        return

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)

    # Export model
    try:
        exported = export(model, args=(normalized_input,))
    except Exception as e:
        print(f"Error exporting model {key}: {e}")
        return

    # Process parameters
    params = exported.state_dict
    new_params = {
        convert_param_name(k): v 
        for k, v in params.items()
    }

    # Generate and save model code
    base_code = exported.graph_module.__str__()
    write_code = apply_templates(base_code)
    
    os.makedirs(model_path, exist_ok=True)
    
    with open(f'{model_path}/model.py', 'w') as fp:
        fp.write(write_code)
    
    # Save metadata
    metadata = {
        "framework": "torch",
        "num_devices_required": 1,
        "num_nodes_required": 1
    }
    
    with open(f'{model_path}/attribute.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    # Save tensor metadata and constraints
    converted = utils.convert_state_and_inputs(params, exported.example_inputs[0])
    utils.save_converted_to_text(
        converted, 
        file_path=f'{model_path}/source_tensor_meta.py'
    )
    utils.save_constraints_text(
        converted, 
        file_path=f'{model_path}/input_tensor_constraints.py'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export torchvision models to txt"
    )
    parser.add_argument(
        "--key", 
        type=str, 
        required=True,
        help="Model name from torchvision.models"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Directory to save the exported model"
    )
    args = parser.parse_args()
    main(key=args.key, model_path=args.model_path)