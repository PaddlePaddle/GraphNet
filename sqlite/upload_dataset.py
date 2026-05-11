import argparse
import os
from datasets import Dataset
from huggingface_hub import HfApi, login


def is_clean_file(filename, root):
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".pyc", ".pyo", ".pyd", ".so"}:
        return False
    if any(x in root for x in ["__pycache__", ".git", ".ipynb_checkpoints"]):
        return False
    return True


def file_generator(base_dir, folders):
    file_list = [
        (os.path.join(root, f), folder)
        for folder in folders
        if os.path.exists(os.path.join(base_dir, folder))
        for root, _, files in os.walk(os.path.join(base_dir, folder))
        for f in files
        if is_clean_file(f, root)
        and os.path.splitext(f)[1].lower() in {".py", ".json", ".txt", ".yaml", ".md"}
    ]

    return (
        {
            "path": os.path.relpath(fp, base_dir),
            "content": open(fp, "r", encoding="utf-8", errors="ignore").read(),
            "source_folder": src,
        }
        for fp, src in file_list
    )


def main(args):
    folders = ["full_graph", "fusible_graph", "sole_op_graph", "typical_graph"]

    login(token=args.hf_token)

    ds = Dataset.from_generator(lambda: file_generator(args.base_dir, folders))
    ds.push_to_hub(
        args.repo_id,
        split=args.split,
        max_shard_size=args.max_shard_size,
        revision=args.revision,
    )
    print("Folder data uploaded successfully!")

    api = HfApi()
    db_path = os.path.join(args.base_dir, args.db_file)
    if os.path.exists(db_path):
        api.upload_file(
            path_or_fileobj=db_path,
            path_in_repo=args.db_file,
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
        )
        print(f"{args.db_file} uploaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload dataset and DB to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf_token", type=str, required=True, help="HuggingFace API token"
    )
    parser.add_argument(
        "--repo_id", type=str, default="PaddlePaddle/GraphNet", help="HF repo ID"
    )
    parser.add_argument(
        "--revision", type=str, default="main", help="HF repo revision/branch"
    )
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Local dataset root directory"
    )
    parser.add_argument(
        "--db_file", type=str, default="GraphNet.db", help="DB filename in base_dir"
    )
    parser.add_argument(
        "--split", type=str, default="GraphNet", help="Dataset split name"
    )
    parser.add_argument(
        "--max_shard_size", type=str, default="500MB", help="Max shard size"
    )
    args = parser.parse_args()
    main(args)
