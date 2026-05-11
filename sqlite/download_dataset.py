import argparse
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def main(args):
    ds = load_dataset(args.repo_id, split=args.split, revision=args.revision)
    for item in ds:
        full_path = os.path.join(args.save_dir, item["path"])
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(item["content"])

    hf_hub_download(
        repo_id=args.repo_id,
        filename=args.db_file,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset and DB from HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id", type=str, default="PaddlePaddle/GraphNet", help="HF repo ID"
    )
    parser.add_argument(
        "--revision", type=str, default="main", help="HF repo revision/branch"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./workspace", help="Local save directory"
    )
    parser.add_argument(
        "--split", type=str, default="GraphNet", help="Dataset split name"
    )
    parser.add_argument(
        "--db_file", type=str, default="GraphNet.db", help="DB filename to download"
    )
    args = parser.parse_args()
    main(args)
