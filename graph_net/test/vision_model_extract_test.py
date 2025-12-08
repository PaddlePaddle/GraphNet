import argparse
import os

import torch
from torchvision import transforms

import graph_net

EXAMPLE_SAMPLE_REL_MODEL_PATHS = [
    "samples/torchvision/wide_resnet50_2",
    "samples/torchvision/wide_resnet101_2",
]


def extract_visio_graph(model_name: str, model_path: str):
    # Normalization parameters for ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Create dummy input
    batch_size = 1
    height, width = 224, 224  # Standard ImageNet size
    num_channels = 3
    random_input = torch.rand(batch_size, num_channels, height, width)
    normalized_input = normalize(random_input)

    # download models using `torchvision.get_model`
    # all_models = list_models(module=torchvision.models)
    # if(model_path not in all_models):
    #     print("Model not found")
    #     return
    # model = torchvision.get_model(model_path, weights="DEFAULT")

    # download models using torch.hub
    # Refer to https://docs.pytorch.org/docs/stable/hub.html
    torch.hub.set_dir(
        "../../../test"
    )  # The default cache directory is $TORCH_HOME/hub; if the environment variable is not set, it defaults to ~/.cache
    endpoints = torch.hub.list("pytorch/vision")
    if model_path not in endpoints:
        print("Model not found")
        return
    model = torch.hub.load("pytorch/vision", model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)

    model = graph_net.torch.extract(name=model_name, dynamic=True)(model)

    print("Running inference...")
    print("Input shape:", normalized_input.shape)
    output = model(normalized_input)
    print("Inference finished. Output shape:", output.shape)


if __name__ == "__main__":
    # get parameters from command line
    workspace_default = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "../../workspace")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="resnet18"
    )  #  Model name (customizable, recommended to be the same as the official name or an abbreviation)
    parser.add_argument(
        "--model_path", type=str, default="resnet18"
    )  # Model name as defined on the official website
    parser.add_argument("--workspace", type=str, default=workspace_default)
    args = parser.parse_args()

    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace

    extract_visio_graph(args.model_name, args.model_path)
