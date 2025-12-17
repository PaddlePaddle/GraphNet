import os
import logging
from typing import Optional

class HFFetcher:
    def __init__(self, workspace: str, token: Optional[str] = None):
        self.workspace = workspace
        self.token = token
        self.logger = logging.getLogger("HFFetcher")

    def download(self, model_id: str) -> str:
        """
        Download model snapshot from Hugging Face.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("huggingface_hub is not installed. Please install it with `pip install huggingface_hub`.")

        self.logger.info(f"Downloading {model_id} from Hugging Face...")
        
        # Define local dir based on model_id
        local_dir = os.path.join(self.workspace, "downloads", model_id.replace("/", "_"))
        
        # Download only necessary files to save time/bandwidth
        allow_patterns = ["*.json", "*.bin", "*.safetensors", "*.py", "*.txt", "*.md"]
        
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=self.token,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False  # Copy files for easier manipulation
        )
        
        return local_dir
