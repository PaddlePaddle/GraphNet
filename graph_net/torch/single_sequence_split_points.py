import argparse
import json
from pathlib import Path


def create_single_operator_ranges(
    op_names_path_prefix,
    model_list,
    output_dir,
    subgraph_ranges_file_name="subgraph_ranges.json",
    subgraph_ranges_json=None,
    output_json=None,
):
    with open(model_list, "r") as f:
        rel_model_paths = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    split_positions_json = {}
    subgraph_ranges_json_data = {}
    for rel_model_path in rel_model_paths:
        txt_path = Path(op_names_path_prefix) / rel_model_path / "op_names.txt"
        if not txt_path.exists():
            print(f"File not found: {txt_path}")
            continue

        with open(txt_path, "r") as f:
            seq = [line.strip() for line in f if line.strip()]

        if not seq:
            print(f"Empty sequence in: {txt_path}")
            continue

        num_ops = len(seq)
        model_name = Path(rel_model_path).name
        subgraph_ranges = [[i, i + 1] for i in range(num_ops)]
        model_output_dir = Path(output_dir) / rel_model_path
        model_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = model_output_dir / subgraph_ranges_file_name
        with open(output_file, "w") as f:
            json.dump({"subgraph_ranges": subgraph_ranges}, f, indent=4)

        split_positions = sorted(
            set(pos for start, end in subgraph_ranges for pos in (start, end))
        )
        split_positions_json[str(rel_model_path)] = {
            "model_name": model_name,
            "split_positions": split_positions,
            "total_length": num_ops,
        }
        subgraph_ranges_json_data[str(rel_model_path)] = {
            "model_name": model_name,
            "subgraph_ranges": subgraph_ranges,
            "total_length": num_ops,
        }
        print(
            f"Created single operator ranges for {model_name}: {len(subgraph_ranges)} subgraphs"
        )

    if output_json:
        with open(output_json, "w") as f:
            json.dump(split_positions_json, f, indent=4)

    if subgraph_ranges_json:
        with open(subgraph_ranges_json, "w") as f:
            json.dump(subgraph_ranges_json_data, f, indent=4)

    return split_positions_json, subgraph_ranges_json_data


def main():
    parser = argparse.ArgumentParser(
        description="Create single operator subgraph ranges"
    )
    parser.add_argument(
        "--model-list", type=str, required=True, help="Path to model list file"
    )
    parser.add_argument(
        "--op-names-path-prefix",
        type=str,
        default="./",
        help="Prefix for op_names.txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for subgraph ranges",
    )
    parser.add_argument(
        "--subgraph-ranges-file-name",
        type=str,
        default="subgraph_ranges.json",
        help="Name of subgraph ranges file",
    )
    parser.add_argument(
        "--subgraph-ranges-json", type=str, help="Path to save combined subgraph ranges"
    )
    parser.add_argument("--output-json", type=str, help="Path to save analysis results")

    args = parser.parse_args()

    create_single_operator_ranges(
        args.op_names_path_prefix,
        args.model_list,
        args.output_dir,
        args.subgraph_ranges_file_name,
        args.subgraph_ranges_json,
        args.output_json,
    )


if __name__ == "__main__":
    main()
