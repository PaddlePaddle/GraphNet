#!/usr/bin/env python3
"""Batch test script for model extraction success rate"""

import argparse
import os
import sys
from datetime import datetime

from graph_net.agent import GraphNetAgent
from graph_net.agent.tests.test_batch_success_rate import (
    get_models_from_hf,
    run_batch_test,
    HUGGINGFACE_HUB_AVAILABLE,
)

# Task distribution ratios for mixed task testing
TEXT_CLASSIFICATION_RATIO = 0.4
TEXT_GENERATION_RATIO = 0.4
QUESTION_ANSWERING_RATIO = 0.2


def main():
    """Run batch test"""

    parser = argparse.ArgumentParser(
        description="Batch test model extraction success rate"
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Number of models to test (default: 100)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="HuggingFace task type (default: None, mixed tasks)",
    )

    args = parser.parse_args()
    model_count = args.count

    print("=" * 70)
    print(f"[START] GraphNet Agent Batch Test - {model_count} models")
    print("=" * 70)

    workspace = os.getenv("GRAPH_NET_EXTRACT_WORKSPACE")
    if not workspace:
        print("\n[ERROR] GRAPH_NET_EXTRACT_WORKSPACE environment variable not set")
        print("\nPlease set workspace:")
        print("  export GRAPH_NET_EXTRACT_WORKSPACE=/path/to/workspace")
        sys.exit(1)

    print(f"\n[INFO] Workspace: {workspace}")

    if not HUGGINGFACE_HUB_AVAILABLE:
        print("\n[ERROR] huggingface_hub not installed")
        print("Please install: pip install huggingface_hub")
        sys.exit(1)

    print(f"\n[INFO] Fetching {model_count} models from HuggingFace Hub...")

    if args.task:
        print(f"   Task type: {args.task}")
    else:
        print("   Task type: Mixed NLP tasks")
    print("   This may take some time...\n")

    try:
        model_ids = []

        if args.task:
            print(f"   - Fetching {args.task} models...")
            models = get_models_from_hf(task=args.task, limit=model_count)
            model_ids.extend(models)
            print(f"     Fetched {len(models)} models")
        else:
            tasks = [
                ("text-classification", int(model_count * TEXT_CLASSIFICATION_RATIO)),
                ("text-generation", int(model_count * TEXT_GENERATION_RATIO)),
                ("question-answering", int(model_count * QUESTION_ANSWERING_RATIO)),
            ]

            for task, limit in tasks:
                print(f"   - Fetching {task} models...")
                models = get_models_from_hf(task=task, limit=limit)
                model_ids.extend(models)
                print(f"     Fetched {len(models)} models")

            model_ids = list(dict.fromkeys(model_ids))

            if len(model_ids) < model_count:
                print(f"   - Current: {len(model_ids)} models, fetching more...")
                additional = get_models_from_hf(
                    task=None, limit=model_count - len(model_ids)
                )
                model_ids.extend(additional)
                model_ids = list(dict.fromkeys(model_ids))

        model_ids = model_ids[:model_count]

        print(f"\n[OK] Successfully fetched {len(model_ids)} models")

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch model list: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n[INIT] Initializing Agent...")
    try:
        agent = GraphNetAgent(workspace=workspace, hf_token=None)
        print("[OK] Agent initialized successfully\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Agent: {e}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"batch_test_{model_count}_models_{timestamp}.json"

    print("=" * 70)
    print("[INFO] Starting batch test")
    print(f"   Model count: {len(model_ids)}")
    print(f"   Output file: {output_file}")
    print("=" * 70)
    print("\n[TIP] Notes:")
    print("   - Test may take hours or longer")
    print("   - Press Ctrl+C to interrupt anytime")
    print("   - Results are saved to JSON file in real-time")
    print("   - Recommended to run with screen or nohup")
    print("\n" + "=" * 70 + "\n")

    try:
        results = run_batch_test(
            agent=agent,
            model_ids=model_ids,
            output_file=output_file,
        )

        print("\n" + "=" * 70)
        print("[DONE] Test completed")
        print("=" * 70)
        print(f"Total models: {results['total']}")
        print(f"Success: {results['success']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {results['success_rate']:.2f}%")
        print(f"Output file: {output_file}")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print("\n\n[WARNING] Test interrupted by user")
        print(f"Partial results saved to: {output_file}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Error during test: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
