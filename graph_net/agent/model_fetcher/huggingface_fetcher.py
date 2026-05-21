"""HuggingFace model fetcher implementation"""

import os
import signal
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    HfApi = None
    snapshot_download = None

from graph_net.agent.model_fetcher.base import BaseModelFetcher
from graph_net.agent.utils.exceptions import (
    GraphExtractionErrorCategory,
    ModelFetchError,
)


class _DownloadTimeoutError(Exception):
    """Raised when snapshot_download exceeds the overall time budget."""

    pass


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
    LocalEntryNotFoundError = None

try:
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )
except ImportError:
    GatedRepoError = None
    HfHubHTTPError = None
    RepositoryNotFoundError = None


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

    def check_accessible(self, model_id: str) -> None:
        """Check whether a HuggingFace model repo is reachable without downloading files."""
        if HfApi is None:
            raise ModelFetchError(
                "huggingface_hub is not installed. "
                "Please install it with: pip install huggingface_hub"
            )

        try:
            if self.endpoint:
                os.environ["HF_ENDPOINT"] = self.endpoint
            api = HfApi(endpoint=self.endpoint)
            api.model_info(
                repo_id=model_id,
                token=self.token,
                files_metadata=False,
            )
        except Exception as e:
            error_category = self._classify_hf_error(e)
            raise ModelFetchError(
                f"Model repo is not accessible for {model_id}: {e}",
                error_category=error_category,
            ) from e

    @staticmethod
    def _classify_hf_error(error: Exception) -> GraphExtractionErrorCategory:
        """Classify HuggingFace API/download errors into extraction categories."""
        if RepositoryNotFoundError is not None and isinstance(
            error, RepositoryNotFoundError
        ):
            return GraphExtractionErrorCategory.MODEL_NOT_FOUND
        if GatedRepoError is not None and isinstance(error, GatedRepoError):
            return GraphExtractionErrorCategory.MODEL_FORBIDDEN
        if HfHubHTTPError is not None and isinstance(error, HfHubHTTPError):
            status_code = getattr(getattr(error, "response", None), "status_code", None)
            if status_code == 404:
                return GraphExtractionErrorCategory.MODEL_NOT_FOUND
            if status_code in (401, 403):
                return GraphExtractionErrorCategory.MODEL_FORBIDDEN

        err_text = str(error)
        if "404 Client Error" in err_text:
            return GraphExtractionErrorCategory.MODEL_NOT_FOUND
        if "401 Client Error" in err_text or "403 Client Error" in err_text:
            return GraphExtractionErrorCategory.MODEL_FORBIDDEN
        return GraphExtractionErrorCategory.MODEL_DOWNLOAD_ERROR

    def download(self, model_id: str) -> Path:
        """
        Download model from HuggingFace Hub with retry on network errors.

        A **hard overall timeout** (signal.alarm) guards `snapshot_download` so that
        if a single file hangs indefinitely (e.g. TCP connection stays open but no
        data arrives), the call is aborted instead of blocking forever.

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

        # Set a stricter download timeout to avoid getting stuck on large/slow files
        # (default is 10s; we bump to 30s to accommodate slow networks while still
        # preventing indefinite hangs).
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get(
            "HF_HUB_DOWNLOAD_TIMEOUT", "30"
        )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # Set endpoint for this call if configured
                if self.endpoint:
                    os.environ["HF_ENDPOINT"] = self.endpoint

                # Hard overall timeout for the entire snapshot_download call.
                # HF_HUB_DOWNLOAD_TIMEOUT=30 only controls individual HTTP requests;
                # huggingface_hub's internal retry/resume logic can still loop forever
                # when a connection stalls without raising an exception.  We therefore
                # enforce a 120-second wall-clock ceiling on the whole operation.
                def _alarm_handler(signum, frame):
                    raise _DownloadTimeoutError(
                        f"Overall download timeout (120s) exceeded for {model_id}"
                    )

                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(120)
                try:
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
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

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
            except _DownloadTimeoutError as e:
                raise ModelFetchError(
                    f"Failed to download model {model_id}: {e}"
                ) from e
            except Exception as e:
                raise ModelFetchError(
                    f"Failed to download model {model_id}: {e}",
                    error_category=self._classify_hf_error(e),
                ) from e

        # Should not reach here, but just in case
        raise ModelFetchError(
            f"Failed to download model {model_id} after {self.max_retries} retries: {last_err}"
        )
