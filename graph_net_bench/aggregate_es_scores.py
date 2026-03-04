import json
import argparse
import numpy as np


def main(args):
    with open(args.es_scores_json_file_path) as f:
        es_scores = json.load(f)

    es_scores = {int(k): v for k, v in es_scores.items()}

    weights = get_weights()
    assert set(weights.keys()) == set(
        es_scores.keys()
    ), f"{set(weights.keys())=}, {set(es_scores.keys())=}"
    weighted_sum = sum(
        weight * np.log(score) / np.log(10)
        for tolerance in weights.keys()
        for weight in [weights[tolerance]]
        for score in [es_scores[tolerance]]
    )
    result = {
        "id": args.sample_id,
        "score": float(weighted_sum),
    }
    with open(args.output_json_file_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"{weighted_sum=}")
    print(f"Result is saved to {args.output_json_file_path}")


def get_weights():
    # `weights` is derived from the NLP ES metrics of NVIDIA A100 relative to H20
    weights = {
        -10: np.float64(0.001),
        -9: np.float64(0.001),
        -8: np.float64(0.001),
        -7: np.float64(0.13),
        -6: np.float64(0.40),
        -5: np.float64(0.48),
        -4: np.float64(0.48),
        -3: np.float64(0.48),
        -2: np.float64(0.48),
        -1: np.float64(0.48),
        0: np.float64(0.48),
        1: np.float64(0.48),
        2: np.float64(0.48),
        3: np.float64(0.48),
        4: np.float64(0.48),
    }
    sum_weights = sum(v for k, v in weights.items())
    return dict((k, v / sum_weights) for k, v in weights.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and aggregate ES(t) scores from benchmark results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--es-scores-json-file-path",
        type=str,
        required=True,
        help="ES scores json file path",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        required=True,
        help="Sample Id",
    )
    parser.add_argument(
        "--output-json-file-path",
        type=str,
        required=True,
        help="json file path for saving the aggregated score",
    )
    parser.set_defaults()
    args = parser.parse_args()
    main(args)
