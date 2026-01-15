import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


def b64_to_json(b64str: str) -> Dict[str, Any]:
    return json.loads(base64.b64decode(b64str).decode("utf-8"))


def json_to_b64(obj: Dict[str, Any]) -> str:
    return base64.b64encode(json.dumps(obj).encode("utf-8")).decode("utf-8")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rewrite_path_to_repo_root(path_value: str, repo_root: Path) -> str:
    """
    Rewrite a path that points into GraphNet repo to server-local repo_root.

    Remap any string containing "graph_net/<...>" to "<repo_root>/graph_net/<...>".
    """
    if not isinstance(path_value, str) or not path_value:
        return path_value

    norm = path_value.replace("\\", "/")
    # If it's already under this repo_root, keep it.
    repo_norm = str(repo_root).replace("\\", "/").rstrip("/")
    if norm.startswith(repo_norm + "/"):
        return str(Path(norm))

    marker = "/graph_net/"
    idx = norm.find(marker)
    if idx >= 0:
        rel = norm[idx + 1 :]  # strip leading "/" to make it relative
        return str(repo_root / rel)
    # Try without leading slash (edge cases)
    marker2 = "graph_net/"
    idx2 = norm.find(marker2)
    if idx2 >= 0:
        rel = norm[idx2:]
        return str(repo_root / rel)
    # Not a repo-internal path; leave as-is.
    return path_value


def deep_rewrite_paths(
    obj: Any, repo_root: Path, keys_to_rewrite: Iterable[str]
) -> Any:
    """
    Recursively rewrite known path fields in a nested dict/list structure.
    """
    if isinstance(obj, dict):
        new_d: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in keys_to_rewrite and isinstance(v, str):
                new_d[k] = rewrite_path_to_repo_root(v, repo_root)
            else:
                new_d[k] = deep_rewrite_paths(v, repo_root, keys_to_rewrite)
        return new_d
    if isinstance(obj, list):
        return [deep_rewrite_paths(x, repo_root, keys_to_rewrite) for x in obj]
    return obj


def prepare_decorator_config_for_server(
    decorator_config_b64: str, output_workspace: str, repo_root: Path
) -> str:
    """
    - Decode client config
    - Rewrite repo paths to server-local repo_root
    - Force output_dir = $OUTPUT_WORKSPACE/samples (torch)
    - Re-encode to base64
    """
    cfg = b64_to_json(decorator_config_b64)

    # These keys are used by subgraph_decompose_and_evaluation_step.py (torch/paddle)
    # plus some forward-compatible names.
    keys_to_rewrite = {
        "decorator_path",
        "custom_extractor_path",
        "post_extract_process_path",
        "post_process_path",
        "extractor_path",
    }
    cfg = deep_rewrite_paths(cfg, repo_root=repo_root, keys_to_rewrite=keys_to_rewrite)

    # Force output dir (torch): OUTPUT_WORKSPACE/samples
    forced_output_dir = os.path.join(output_workspace, "samples")
    try:
        custom_cfg = cfg["decorator_config"]["custom_extractor_config"]
        if isinstance(custom_cfg, dict):
            custom_cfg["output_dir"] = forced_output_dir
    except Exception:
        pass

    return json_to_b64(cfg)


def run_remote_subgraph_decompose(args) -> int:
    input_workspace = os.environ.get("INPUT_WORKSPACE") or args.model_path
    output_workspace = os.environ.get("OUTPUT_WORKSPACE")

    if not input_workspace:
        print(
            "[remote_subgraph_decompose_entry][ERROR] Missing INPUT_WORKSPACE and --model-path.",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if not output_workspace:
        print(
            "[remote_subgraph_decompose_entry][ERROR] Missing OUTPUT_WORKSPACE env var.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    input_workspace = os.path.abspath(input_workspace)
    output_workspace = os.path.abspath(output_workspace)
    os.makedirs(output_workspace, exist_ok=True)

    repo_root = get_repo_root()

    # Rewrite decorator-config for server environment
    new_b64 = prepare_decorator_config_for_server(
        decorator_config_b64=args.decorator_config,
        output_workspace=output_workspace,
        repo_root=repo_root,
    )

    # Ensure forced samples dir exists (not required, but clearer)
    os.makedirs(os.path.join(output_workspace, "samples"), exist_ok=True)

    log_path = os.path.join(output_workspace, args.log_file)

    cmd = [
        sys.executable,
        "-m",
        "graph_net.torch.run_model",
        "--model-path",
        input_workspace,
        "--decorator-config",
        new_b64,
    ]

    # Helpful markers that show up in the gRPC server captured stderr/stdout as well.
    print(
        f"[remote_subgraph_decompose_entry] repo_root: {repo_root}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[remote_subgraph_decompose_entry] INPUT_WORKSPACE: {input_workspace}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[remote_subgraph_decompose_entry] OUTPUT_WORKSPACE: {output_workspace}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[remote_subgraph_decompose_entry] log_path: {log_path}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"[remote_subgraph_decompose_entry] cmd: {' '.join(cmd)}",
        file=sys.stderr,
        flush=True,
    )

    # Run and log to OUTPUT_WORKSPACE (will be included in output.tar.gz)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=f, text=True)

    if proc.returncode != 0:
        print(
            f"[remote_subgraph_decompose_entry][ERROR] run_model failed with returncode={proc.returncode}. "
            f"See {args.log_file} in returned output.tar.gz.",
            file=sys.stderr,
            flush=True,
        )

    return proc.returncode


def main(args) -> int:
    return run_remote_subgraph_decompose(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Server entrypoint: remote subgraph decomposition for torch."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default=None,
        help="Optional; if not provided, uses $INPUT_WORKSPACE. Kept for CLI symmetry.",
    )
    parser.add_argument(
        "--decorator-config",
        type=str,
        required=True,
        help="Base64-encoded decorator config JSON from client.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="log_remote_decompose.txt",
        help="Log filename to create under $OUTPUT_WORKSPACE.",
    )
    args = parser.parse_args()
    main(args)
