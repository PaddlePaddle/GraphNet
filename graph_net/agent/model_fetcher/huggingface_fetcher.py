"""HuggingFace model fetcher implementation"""

import os
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

from graph_net.agent.model_fetcher.base import BaseModelFetcher
from graph_net.agent.utils.exceptions import (
    GraphExtractionErrorCategory,
    ModelFetchError,
)

# Network-related exceptions that are worth retrying
_RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# Try to import httpx/huggingface_hub errors for more granular retry
try:
    import httpx

    _RETRYABLE_ERRORS = _RETRYABLE_ERRORS + (httpx.ConnectTimeout, httpx.ReadTimeout)
except ImportError:
    pass

try:
    from huggingface_hub.errors import LocalEntryNotFoundError

    _RETRYABLE_ERRORS = _RETRYABLE_ERRORS + (LocalEntryNotFoundError,)
except ImportError:
    pass


class HFFetcher(BaseModelFetcher):
    """HuggingFace model fetcher using huggingface_hub"""

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 5  # seconds, will be exponentially backed off

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        endpoint: Optional[str] = None,
    ):
        """
        Args:
            cache_dir:   Directory to cache downloaded models
            token:       HuggingFace API token (optional, for private models)
            max_retries: Max retry attempts on network errors (default 3)
            retry_delay: Initial delay between retries in seconds (default 5, exponential backoff)
            endpoint:    HuggingFace mirror endpoint (e.g., "https://hf-mirror.com").
                         If not set, falls back to HF_ENDPOINT env var.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.token = token
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Resolve endpoint: explicit param > env var
        self.endpoint = endpoint or os.environ.get("HF_ENDPOINT")

    def download(self, model_id: str) -> Path:
        """
        Download model from HuggingFace Hub with retry on network errors.

        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")

        Returns:
            Path to local model directory

        Raises:
            ModelFetchError: If download fails after all retries
        """
        if snapshot_download is None:
            raise ModelFetchError(
                "huggingface_hub is not installed. "
                "Please install it with: pip install huggingface_hub"
            )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Set endpoint for this call if configured
                if self.endpoint:
                    os.environ["HF_ENDPOINT"] = self.endpoint

                local_dir = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    token=self.token,
                    ignore_patterns=[
                        "*.bin",
                        "*.safetensors",
                        "*.pt",
                        "*.pth",
                        "*.gguf",
                        "*.ot",
                        "*.zip",
                        "*.tflite",
                        "*.mlmodel",
                        "*.onnx",
                        "*.msgpack",
                        "flax_model*",
                        "tf_model*",
                        "rust_model*",
                    ],
                )
                return Path(local_dir)

            except _RETRYABLE_ERRORS as e:
                last_err = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    # Check if the error message indicates a timeout — these are worth retrying
                    err_msg = str(e).lower()
                    is_timeout = any(
                        kw in err_msg
                        for kw in ("timeout", "timed out", "connection", "refused")
                    )
                    if not is_timeout:
                        raise ModelFetchError(
                            f"Failed to download model {model_id}: {e}"
                        ) from e

                    print(
                        f"[HFFetcher] Network error for {model_id} "
                        f"(attempt {attempt}/{self.max_retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    raise ModelFetchError(
                        f"Failed to download model {model_id} after {self.max_retries} retries: {e}"
                    ) from e
            except Exception as e:
                err_text = str(e)
                if "404 Client Error" in err_text:
                    raise ModelFetchError(
                        f"Failed to download model {model_id}: {e}",
                        error_category=GraphExtractionErrorCategory.MODEL_NOT_FOUND,
                    ) from e
                if "403 Client Error" in err_text:
                    raise ModelFetchError(
                        f"Failed to download model {model_id}: {e}",
                        error_category=GraphExtractionErrorCategory.MODEL_FORBIDDEN,
                    ) from e
                raise ModelFetchError(
                    f"Failed to download model {model_id}: {e}"
                ) from e

        # Should not reach here, but just in case
        raise ModelFetchError(
            f"Failed to download model {model_id} after {self.max_retries} retries: {last_err}"
        )
