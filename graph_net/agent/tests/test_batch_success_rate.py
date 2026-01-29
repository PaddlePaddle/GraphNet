"""Batch testing script for GraphNet Agent success rate statistics"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from huggingface_hub import list_models

    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

from graph_net.agent import GraphNetAgent

# Default test models (common and small models)
DEFAULT_TEST_MODELS = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "gpt2",
    "t5-small",
    "albert-base-v2",
    "google/bert_uncased_L-2_H-128_A-2",
    "google/t5-efficient-mini",
]


def get_models_from_hf(task: Optional[str] = None, limit: int = 100) -> List[str]:
    """Get model list from HuggingFace Hub"""
    if not HUGGINGFACE_HUB_AVAILABLE:
        print("[WARNING] huggingface_hub not installed, cannot fetch models from Hub")
        return []

    try:
        print(
            f"[INFO] Fetching models from HuggingFace Hub (task={task}, limit={limit})..."
        )

        search_params = {
            "sort": "downloads",
            "direction": -1,  # descending order
            "limit": limit,
        }

        if task:
            search_params["task"] = task

        models = list(list_models(**search_params))
        model_ids = [model.id for model in models]

        print(f"[OK] Fetched {len(model_ids)} models")
        return model_ids

    except Exception as e:
        print(f"[ERROR] Failed to fetch model list: {e}")
        return []


def load_models_from_file(file_path: str) -> List[str]:
    """Load model list from file (one model ID per line)"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            models = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
        print(f"[OK] Loaded {len(models)} models from file: {file_path}")
        return models
    except (OSError, IOError) as e:
        print(f"[ERROR] Failed to load model list: {e}")
        return []


def run_batch_test(
    agent: GraphNetAgent,
    model_ids: List[str],
    output_file: Optional[str] = None,
) -> Dict:
    """Run batch test and calculate success rate"""
    results = {
        "total": len(model_ids),
        "success": 0,
        "failed": 0,
        "success_rate": 0.0,
        "start_time": datetime.now().isoformat(),
        "details": [],
    }

    print(f"\n{'='*60}")
    print(f"[START] Starting batch test: {len(model_ids)} models")
    print(f"{'='*60}\n")

    for idx, model_id in enumerate(model_ids, 1):
        print(f"\n[{idx}/{len(model_ids)}] Testing: {model_id}")
        print("-" * 60)

        start_time = time.time()
        try:
            success = agent.extract_sample(model_id)
            elapsed = time.time() - start_time

            if success:
                results["success"] += 1
                status = "[OK] Success"
            else:
                results["failed"] += 1
                status = "[FAIL] Failed"

            result_entry = {
                "model_id": model_id,
                "success": success,
                "elapsed_time": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            }
            results["details"].append(result_entry)

            print(f"{status} (elapsed: {elapsed:.2f}s)")

        except KeyboardInterrupt:
            print("\n[WARNING] Test interrupted by user")
            break
        except Exception as e:
            elapsed = time.time() - start_time
            results["failed"] += 1
            result_entry = {
                "model_id": model_id,
                "success": False,
                "error": str(e),
                "elapsed_time": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
            }
            results["details"].append(result_entry)
            print(f"[ERROR] Exception: {e} (elapsed: {elapsed:.2f}s)")

        # Show real-time statistics
        current_success_rate = (results["success"] / idx) * 100
        print(
            f"[STATS] Current success rate: {current_success_rate:.2f}% ({results['success']}/{idx})"
        )

    results["end_time"] = datetime.now().isoformat()
    results["success_rate"] = (
        (results["success"] / results["total"]) * 100 if results["total"] > 0 else 0.0
    )

    # Save results
    if output_file:
        _save_results(results, output_file)

    # Print final statistics
    _print_statistics(results)

    return results


def _save_results(results: Dict, output_file: str) -> None:
    """Save test results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] Results saved to: {output_path}")
    except (OSError, IOError) as e:
        print(f"[WARNING] Failed to save results: {e}")


def _print_statistics(results: Dict) -> None:
    """Print final test statistics"""
    print(f"\n{'='*60}")
    print("[SUMMARY] Test Summary")
    print(f"{'='*60}")
    print(f"Total models: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success_rate']:.2f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch test GraphNet Agent success rate"
    )
    parser.add_argument(
        "--model-list-file",
        type=str,
        help="Model list file path (one model ID per line)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of models to fetch from HuggingFace Hub (default: 10)",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="HuggingFace task type (e.g., text-classification, text-generation)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace path (default: use GRAPH_NET_EXTRACT_WORKSPACE env var)",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None, help="HuggingFace API Token (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="batch_test_results.json",
        help="Output file path for results (default: batch_test_results.json)",
    )
    parser.add_argument(
        "--use-default-models",
        action="store_true",
        help="Use predefined default test model list",
    )

    args = parser.parse_args()

    workspace = args.workspace or os.getenv("GRAPH_NET_EXTRACT_WORKSPACE")
    if not workspace:
        print("[ERROR] workspace not specified")
        print(
            "   Use --workspace or set GRAPH_NET_EXTRACT_WORKSPACE environment variable"
        )
        sys.exit(1)

    model_ids = _get_model_list(args)
    if not model_ids:
        print("[ERROR] no models to test")
        sys.exit(1)

    agent = _init_agent(workspace, args.hf_token)

    results = run_batch_test(
        agent=agent,
        model_ids=model_ids,
        output_file=args.output,
    )

    sys.exit(0 if results["success_rate"] > 0 else 1)


def _get_model_list(args: argparse.Namespace) -> List[str]:
    """Get model list from various sources"""
    if args.use_default_models:
        print(f"[INFO] Using default model list ({len(DEFAULT_TEST_MODELS)} models)")
        return DEFAULT_TEST_MODELS

    if args.model_list_file:
        return load_models_from_file(args.model_list_file)

    if HUGGINGFACE_HUB_AVAILABLE:
        return get_models_from_hf(task=args.task, limit=args.count)

    print("[WARNING] No model source specified, using default model list")
    return DEFAULT_TEST_MODELS


def _init_agent(workspace: str, hf_token: Optional[str]) -> GraphNetAgent:
    """Initialize GraphNetAgent"""
    print(f"\n[INIT] Initializing Agent (workspace: {workspace})...")
    try:
        agent = GraphNetAgent(workspace=workspace, hf_token=hf_token)
        print("[OK] Agent initialized successfully\n")
        return agent
    except Exception as e:
        print(f"[ERROR] Failed to initialize Agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
