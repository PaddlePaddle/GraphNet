import argparse
import os

import torch
import torchvision
from torchvision import transforms
from torchvision.models import list_models, get_model
import graph_net


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

    # 使用get_model下载模型
    # all_models = list_models(module=torchvision.models)
    # if(model_path not in all_models):
    #     print("不存在该模型, 请校验模型名称是否相同")
    #     return
    # model = get_model(model_path, weights="DEFAULT")

    # 使用torch.hub下载模型
    # 相关使用办法见https://docs.pytorch.org/docs/stable/hub.html
    torch.hub.set_dir("../../../test")  # 缓存目录默认为$TORCH_HOME/hub 如果没有设置环境变量则为 ~/.cache
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
    )  # 模型名称(自定义,推荐与官网相同或者简写)
    parser.add_argument("--model_path", type=str, default="resnet18")  # 官网定义模型名称
    parser.add_argument("--workspace", type=str, default=workspace_default)
    args = parser.parse_args()

    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace

    extract_visio_graph(args.model_name, args.model_path)
