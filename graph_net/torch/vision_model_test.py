import argparse
import os
import json
import torch
import torchvision
from torchvision import transforms
from graph_net.torch._extract.extractor import extract
import os

os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = "../../samples/torch/extracted_models"

if __name__ == '__main__':
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

    # Instantiate model
    model = torchvision.models.get_model("resnet18", weights="DEFAULT")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalized_input = normalized_input.to(device)
    
    model = extract(name="resnet18")(model)

    print("Running inference...")
    print("Input shape:", normalized_input.shape)
    output = model(normalized_input)
    print("Inference finished. Output shape:", output.shape)
