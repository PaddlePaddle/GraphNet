"""HuggingFace model fetcher implementation"""

from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from graph_net.agent.model_fetcher.base import BaseModelFetcher
from graph_net.agent.utils.exceptions import ModelFetchError


class HFFetcher(BaseModelFetcher):
    """HuggingFace model fetcher using huggingface_hub"""

    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        Args:
            cache_dir: Directory to cache downloaded models
            token: HuggingFace API token (optional, for private models)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.token = token

    def download(self, model_id: str) -> Path:
        """
        Download model from HuggingFace Hub

        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")

        Returns:
            Path to local model directory

        Raises:
            ModelFetchError: If download fails
        """
        if snapshot_download is None:
            raise ModelFetchError(
                "huggingface_hub is not installed. "
                "Please install it with: pip install huggingface_hub"
            )

        try:
            # Use snapshot_download to get all model files
            local_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                token=self.token,
            )
            return Path(local_dir)
        except Exception as e:
            raise ModelFetchError(f"Failed to download model {model_id}: {e}") from e
