import argparse
import os

import torch
import graph_net.torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "perplexity-ai/pplx-embed-v1-0.6b"
DEFAULT_GRAPH_NAME = "pplx-embed-v1-0.6b"


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_inputs(
    tokenizer: AutoTokenizer, device: torch.device, max_length: int
) -> dict:
    text = [
        "GraphNet extracts model computation graphs.",
    ]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract computation graph for pplx-embed-v1-0.6b"
    )
    parser.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id"
    )
    parser.add_argument(
        "--graph-name", default=DEFAULT_GRAPH_NAME, help="Output model folder name"
    )
    parser.add_argument(
        "--device", default="auto", help='Device: "auto", "cpu", or e.g. "cuda:0"'
    )
    parser.add_argument(
        "--max-length", type=int, default=64, help="Tokenizer max_length"
    )
    parser.add_argument(
        "--dynamic",
        choices=["true", "false"],
        default="false",
        help="Whether to enable dynamic shape mode in graph extraction",
    )
    args = parser.parse_args()

    workspace = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
    if not workspace:
        raise EnvironmentError(
            "Please export GRAPH_NET_EXTRACT_WORKSPACE before running."
        )

    os.makedirs(workspace, exist_ok=True)

    device = select_device(args.device)
    dtype = select_dtype(device)

    print(f"[Info] workspace={workspace}")
    print(f"[Info] model_id={args.model_id}")
    print(f"[Info] graph_name={args.graph_name}")
    print(f"[Info] device={device}, dtype={dtype}")
    print(f"[Info] dynamic={args.dynamic}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    inputs = build_inputs(tokenizer, device, args.max_length)

    dynamic_mode = args.dynamic == "true"
    wrapped = graph_net.torch.extract(name=args.graph_name, dynamic=dynamic_mode)(
        model
    ).eval()

    print("[Info] running one forward pass for extraction...")
    with torch.no_grad():
        outputs = wrapped(**inputs)

    if isinstance(outputs, torch.Tensor):
        out_shape = tuple(outputs.shape)
    elif hasattr(outputs, "last_hidden_state"):
        out_shape = tuple(outputs.last_hidden_state.shape)
    elif (
        isinstance(outputs, (tuple, list))
        and len(outputs) > 0
        and isinstance(outputs[0], torch.Tensor)
    ):
        out_shape = tuple(outputs[0].shape)
    else:
        out_shape = "unknown"

    print(f"[Info] extraction finished, output_shape={out_shape}")


if __name__ == "__main__":
    main()
