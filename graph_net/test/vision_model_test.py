import argparse
import timm
import os
import json
import torch
import torchvision
from torchvision import transforms
import os
import graph_net
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="efficientnet_v2_s")
    args = parser.parse_args()
    model_name = args.model_name
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

    # Instantiate model
    try:
        model = torchvision.models.get_model(model_name, weights="DEFAULT")
    except:
        print(f"trying find model {model_name} in timm")
        model = timm.create_model(model_name, pretrained=False)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)

<<<<<<< Updated upstream
    model = graph_net.torch.extract(name="resnet18", dynamic=True)(model)
=======
    model = graph_net.torch.extract(name=model_name)(model)
>>>>>>> Stashed changes

    print("Running inference...")
    print("Input shape:", normalized_input.shape)
    output = model(normalized_input)
    print("Inference finished. Output shape:", output.shape)
