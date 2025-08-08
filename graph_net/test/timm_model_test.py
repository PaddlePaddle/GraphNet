# pip install timm

import argparse
import os
import torch
from torchvision import transforms
import timm  # 导入 timm 库
import graph_net

os.environ["TIMMDL_DISABLE_RETRY"] = "1"  # 禁用重试


def extract_visio_graph(model_name, model_path, dynamic_mode):
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

    # Instantiate model using timm
    model = timm.create_model(model_path, pretrained=False)  # 使用 timm 加载 resnet18
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)

    # Extract graph structure
    model = graph_net.torch.extract(name=model_name, dynamic=dynamic_mode)(model)

    print("Running inference...")
    output = model(normalized_input)
    print("Inference finished. Output shape:", output.shape)


if __name__ == "__main__":
    # get parameters from command line
    workspace_default = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "workspace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="resnet18")  # timm 模型名称
    parser.add_argument("--workspace", type=str, default=workspace_default)
    parser.add_argument("--dynamic", type=bool, default=True)
    args = parser.parse_args()

    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace

    extract_visio_graph(args.model_name, args.model_path, args.dynamic)
