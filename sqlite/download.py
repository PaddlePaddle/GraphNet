import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ID = "PaddlePaddle/GraphNet"
REVISION = "20260224"
SAVE_DIR = "./workspace"

ds = load_dataset(REPO_ID, split="GraphNet", revision=REVISION)
for item in ds:
    full_path = os.path.join(SAVE_DIR, item["path"])
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(item["content"])

hf_hub_download(
    repo_id=REPO_ID,
    filename="GraphNet.db",
    repo_type="dataset",
    revision=REVISION,
    local_dir=SAVE_DIR,
)
