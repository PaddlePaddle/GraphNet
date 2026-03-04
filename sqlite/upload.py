import os
from datasets import Dataset
from huggingface_hub import HfApi, login


HF_TOKEN = "<KEY>"
REPO_ID = "PaddlePaddle/GraphNet"
REVISION = "20260203"
BASE_DIR = "/work/GraphNet/torch_paddle_samples/subgraph_dataset_20260203"
FOLDERS_TO_PACK = ["full_graph", "fusible_graph", "sole_op_graph", "typical_graph"]
DB_FILE = "GraphNet.db"


def is_clean_file(filename, root):
    ext = os.path.splitext(filename)[1].lower()
    if ext in {".pyc", ".pyo", ".pyd", ".so"}:
        return False
    if any(x in root for x in ["__pycache__", ".git", ".ipynb_checkpoints"]):
        return False
    return True


def file_generator():
    file_list = [
        (os.path.join(root, f), folder)
        for folder in FOLDERS_TO_PACK
        if os.path.exists(os.path.join(BASE_DIR, folder))
        for root, _, files in os.walk(os.path.join(BASE_DIR, folder))
        for f in files
        if is_clean_file(f, root)
        and os.path.splitext(f)[1].lower() in {".py", ".json", ".txt", ".yaml", ".md"}
    ]

    return (
        {
            "path": os.path.relpath(fp, BASE_DIR),
            "content": open(fp, "r", encoding="utf-8", errors="ignore").read(),
            "source_folder": src,
        }
        for fp, src in file_list
    )


def main():
    login(token=HF_TOKEN)

    ds = Dataset.from_generator(file_generator)
    ds.push_to_hub(REPO_ID, split="GraphNet", max_shard_size="500MB", revision=REVISION)
    print("Folder data uploaded successfully!")

    api = HfApi()
    db_path = os.path.join(BASE_DIR, DB_FILE)
    if os.path.exists(db_path):
        api.upload_file(
            path_or_fileobj=db_path,
            path_in_repo=DB_FILE,
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
        )
        print(f"{DB_FILE} uploaded successfully!")


if __name__ == "__main__":
    main()
