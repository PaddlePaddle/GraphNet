import argparse
import os

import torch
from torchvision import transforms
from torchvision.models import get_model, get_model_weights

import graph_net


def extract_swin_t_graph(model_name: str, model_path: str):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    batch_size = 1
    height, width = 224, 224
    num_channels = 3
    random_input = torch.rand(batch_size, num_channels, height, width)
    normalized_input = normalize(random_input)

    weights = None
    try:
        w = get_model_weights(model_path)
        weights = w.DEFAULT
    except Exception:
        pass

    model = get_model(model_path, weights=weights)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)

    model = graph_net.torch.extract(name=model_name, dynamic=False)(model)

    print("Running inference...")
    print("Input shape:", normalized_input.shape)
    with torch.no_grad():
        output = model(normalized_input)
    print("Inference finished. Output shape:", output.shape)


if __name__ == "__main__":
    workspace_default = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "workspace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="swin_t")
    parser.add_argument("--model_path", type=str, default="swin_t")
    parser.add_argument("--workspace", type=str, default=workspace_default)
    args = parser.parse_args()

    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace

    extract_swin_t_graph(args.model_name, args.model_path)
