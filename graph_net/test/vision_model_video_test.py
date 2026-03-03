import argparse
import os
import torch
import torchvision
from torchvision import transforms
import graph_net
import importlib

def load_video_model(model_name):
    """
    动态加载视频模型，兼容不同版本的 torchvision
    """
    video_module = importlib.import_module("torchvision.models.video")
    if hasattr(video_module, model_name):
        model_builder = getattr(video_module, model_name)
        if isinstance(model_builder, type):
            try:
                weights_enum = getattr(torchvision.models.video, f"{model_name.upper()}_WEIGHTS", None)
                if weights_enum and hasattr(weights_enum, "DEFAULT"):
                    return model_builder(weights=weights_enum.DEFAULT)
                else:
                    return model_builder(pretrained=True)
            except Exception as e:
                print(f"Initialize randomly")
                return model_builder()
        else:
            # 如果是函数，直接调用
            return model_builder(pretrained=True)


    raise ValueError(f"Not found: {model_name}")


def extract_visio_graph(model_name, model_path):
    model = load_video_model(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size, channels, frames, height, width = 1, 3, 16, 224, 224
    random_input = torch.rand(batch_size, channels, frames, height, width)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, channels, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, channels, 1, 1, 1)
    normalized_input = (random_input - mean) / std

    normalized_input = normalized_input.to(device)

    model = graph_net.torch.extract(name=model_name, dynamic=True)(model)

    print("Running inference...")
    print("Input shape:", normalized_input.shape)

    output = model(normalized_input)



if __name__ == "__main__":
    workspace_default = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "../../workspace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mc3_18")
    parser.add_argument("--model_path", type=str, default="mc3_18")
    parser.add_argument("--workspace", type=str, default=workspace_default)
    parser.add_argument("--dynamic", type=bool, default=True)
    args = parser.parse_args()

    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = args.workspace

    extract_visio_graph(args.model_name, args.model_path)