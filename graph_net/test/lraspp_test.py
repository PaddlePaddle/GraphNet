import argparse
import os

import torch
import torchvision

import graph_net


def extract_lraspp_graph(model_name: str, workspace: str) -> None:
    """Extract computation graph for torchvision's lraspp_mobilenet_v3_large."""
    os.environ["GRAPH_NET_EXTRACT_WORKSPACE"] = workspace
    os.makedirs(workspace, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model: {model_name} on {device}")

    # 1. Instantiate model with default pretrained weights.
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights="DEFAULT"
    )
    model = model.to(device).eval()

    # 2. Build dummy input. lraspp_mobilenet_v3_large accepts a single (B,C,H,W)
    #    RGB tensor; 520x520 matches the COCO->VOC evaluation recipe.
    batch_size, channels, height, width = 1, 3, 520, 520
    input_data = torch.rand(batch_size, channels, height, width, device=device)

    # 3. Wrap with extractor and run once to capture the graph.
    #    LRASPP internally calls F.interpolate(x, size=input.shape[-2:]),
    #    which produces a torch.Size with symbolic ints under dynamic=True
    #    and breaks the graph. Use static shape tracing instead.
    wrapped = graph_net.torch.extract(name=model_name, dynamic=False)(model)

    print("Running inference...")
    print("Input shape:", tuple(input_data.shape))
    with torch.no_grad():
        output = wrapped(input_data)

    out_tensor = output["out"] if isinstance(output, dict) else output
    print("Inference finished. Output shape:", tuple(out_tensor.shape))
    print(f"[OK] {model_name} graph extraction succeeded.")


if __name__ == "__main__":
    workspace_default = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE", "workspace")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="lraspp_mobilenet_v3_large",
        help="Name used as the extracted sample directory name.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=workspace_default,
        help="Directory to write the extracted graph artifacts to.",
    )
    args = parser.parse_args()

    extract_lraspp_graph(args.model_name, args.workspace)
