import argparse
import torch
import torchvision
from torchvision.models import get_model, get_model_weights, list_models
from graph_net.torch.extractor import extract


def run_model(name: str, device: torch.device) -> None:
    print(f"\nTesting model: {name}")
    weights = None
    try:
        w = get_model_weights(name)
        weights = w.DEFAULT
    except Exception:
        pass
    try:
        model = get_model(name, weights=weights)
    except Exception as e:
        print(f"[FAIL] {name}: failed to instantiate model - {e}")
        return

    cfg = getattr(model, "default_cfg", {})
    input_size = cfg.get("input_size", (3, 224, 224))
    # Video models
    if "r3d" in name or "mvit" in name:
        t = cfg.get("num_frames", 16)
        input_shape = (input_size[0], t, input_size[1], input_size[2])
    else:
        input_shape = input_size

    module_name = model.__class__.__module__
    is_detection = module_name.startswith("torchvision.models.detection")
    is_segmentation = module_name.startswith("torchvision.models.segmentation")

    if is_detection or is_segmentation:
        input_data = [torch.rand(input_shape)]
    else:
        input_data = torch.rand(1, *input_shape)

    model.to(device)
    if isinstance(input_data, list):
        input_data = [t.to(device) for t in input_data]
    else:
        input_data = input_data.to(device)

    wrapped = extract(name=name)(model)
    wrapped.eval()

    try:
        with torch.no_grad():
            wrapped(input_data)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.models:
        models = args.models
    else:
        models = list_models()

    for name in models[:35]:
        run_model(name, device)
    for name in models[38:]:
        run_model(name, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run torchvision models with GraphNet extractor"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Model names to test. Defaults to all models returned by torchvision",
    )
    args = parser.parse_args()
    main(args)
