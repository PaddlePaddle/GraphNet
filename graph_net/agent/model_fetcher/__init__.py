"""Model fetching modules"""

from graph_net.agent.model_fetcher.base import BaseModelFetcher
from graph_net.agent.model_fetcher.huggingface_fetcher import HFFetcher

__all__ = ["BaseModelFetcher", "HFFetcher"]
