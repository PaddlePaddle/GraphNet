"""Subprocess-based graph extractor implementation"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from graph_net.agent.graph_extractor.base import BaseGraphExtractor
from graph_net.agent.utils.exceptions import ExtractionError

# Constants
DEFAULT_TIMEOUT = 600  # 10 minutes for large models
OUTPUT_SEARCH_WINDOW = 600  # 10 minutes for finding recently created directories
HASH_DIR_LENGTH = 40  # SHA1 hash length
ERROR_MSG_MAX_LINES = 20  # Keep first and last N lines of error messages


class SubprocessGraphExtractor(BaseGraphExtractor):
    """Extractor that runs script in subprocess"""

    def __init__(self, workspace: str, timeout: int = DEFAULT_TIMEOUT):
        """
        Args:
            workspace: Workspace root directory
            timeout: Timeout in seconds for script execution
        """
        self.workspace = Path(workspace)
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, code_path: Path, model_id: str) -> Path:
        """
        Execute script and extract computation graph

        Args:
            code_path: Path to run_model.py script
            model_id: Model ID for output directory naming

        Returns:
            Path to extracted sample directory

        Raises:
            ExtractionError: If extraction fails
        """
        try:
            # Get GraphNet root directory for PYTHONPATH
            from graph_net.graph_net_root import get_graphnet_root

            graphnet_root = get_graphnet_root()

            # Prepare environment with PYTHONPATH
            env = os.environ.copy()
            current_pythonpath = env.get("PYTHONPATH", "")
            if current_pythonpath:
                env["PYTHONPATH"] = f"{graphnet_root}:{current_pythonpath}"
            else:
                env["PYTHONPATH"] = str(graphnet_root)

            # Run script in subprocess
            result = subprocess.run(
                [sys.executable, str(code_path)],
                cwd=str(code_path.parent),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                error_msg = self._format_error_message(result.stderr or result.stdout)
                raise ExtractionError(
                    f"Script execution failed with return code {result.returncode}.\n"
                    f"Command: {sys.executable} {code_path}\n"
                    f"Error output:\n{error_msg}"
                )

            # Find output directory using multiple strategies
            output_dir = self._find_output_dir_robust(model_id)

            if not output_dir or not output_dir.exists():
                raise ExtractionError(
                    f"Output directory not found for model: {model_id}.\n"
                    f"Searched in workspace: {self.workspace}\n"
                    f"Please check if the extraction script executed successfully."
                )

            return output_dir
        except subprocess.TimeoutExpired:
            raise ExtractionError(
                f"Script execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            raise ExtractionError(f"Failed to extract graph: {e}") from e

    def _format_error_message(self, error_msg: str) -> str:
        """Format error message, truncating if too long"""
        if len(error_msg) <= 2000:
            return error_msg

        error_lines = error_msg.split("\n")
        if len(error_lines) <= ERROR_MSG_MAX_LINES * 2:
            return error_msg

        truncated = "\n".join(
            error_lines[:ERROR_MSG_MAX_LINES]
            + ["... (truncated) ..."]
            + error_lines[-ERROR_MSG_MAX_LINES:]
        )
        return truncated

    def _find_output_dir_robust(self, model_id: str) -> Optional[Path]:
        """Robustly find output directory for extracted sample"""
        workspace_path = self._get_workspace_path()
        if not workspace_path or not workspace_path.exists():
            self.logger.warning(f"Workspace path does not exist: {workspace_path}")
            return None

        safe_model_id = model_id.replace("/", "_")
        expected_dir = workspace_path / safe_model_id

        # Strategy 1: Expected directory name
        if expected_dir.exists() and self._is_valid_sample_dir(expected_dir):
            self.logger.info(f"Found output directory: {expected_dir}")
            return expected_dir

        # Strategy 2: Retry after file system sync
        time.sleep(1)
        if expected_dir.exists() and self._is_valid_sample_dir(expected_dir):
            self.logger.info(f"Found output directory (retry): {expected_dir}")
            return expected_dir

        # Strategy 3: Search for recently modified directories
        recent_dir = self._find_recent_sample_dir(workspace_path)
        if recent_dir:
            return recent_dir

        # Strategy 4: Search by model_id pattern
        pattern_dir = self._find_dir_by_pattern(workspace_path, model_id, safe_model_id)
        if pattern_dir:
            return pattern_dir

        # Strategy 5: Search for hash-named directories
        hash_dir = self._find_hash_named_dir(workspace_path)
        if hash_dir:
            return hash_dir

        self.logger.warning(f"Could not find output directory for model: {model_id}")
        return None

    def _get_workspace_path(self) -> Optional[Path]:
        """Get workspace path from environment or instance variable"""
        workspace_env = os.environ.get("GRAPH_NET_EXTRACT_WORKSPACE")
        return Path(workspace_env) if workspace_env else self.workspace

    def _find_recent_sample_dir(self, workspace_path: Path) -> Optional[Path]:
        """Find most recently modified sample directory"""
        current_time = time.time()
        candidate_dirs = []

        for item in workspace_path.iterdir():
            if not item.is_dir() or not self._is_valid_sample_dir(item):
                continue
            try:
                mtime = item.stat().st_mtime
                time_diff = current_time - mtime
                if time_diff < OUTPUT_SEARCH_WINDOW:
                    candidate_dirs.append((item, mtime, time_diff))
            except (OSError, FileNotFoundError):
                continue

        if candidate_dirs:
            candidate_dirs.sort(key=lambda x: x[1], reverse=True)
            most_recent = candidate_dirs[0][0]
            time_diff = candidate_dirs[0][2]
            self.logger.info(
                f"Found recent directory: {most_recent} (modified {time_diff:.1f}s ago)"
            )
            return most_recent

        return None

    def _find_dir_by_pattern(
        self, workspace_path: Path, model_id: str, safe_model_id: str
    ) -> Optional[Path]:
        """Find directory matching model_id patterns"""
        patterns = [
            safe_model_id,
            model_id.replace("/", "-"),
            model_id.replace("/", "_"),
        ]

        for item in workspace_path.iterdir():
            if not item.is_dir():
                continue
            if any(pattern in item.name for pattern in patterns):
                if self._is_valid_sample_dir(item):
                    self.logger.info(f"Found directory by pattern: {item}")
                    return item
        return None

    def _find_hash_named_dir(self, workspace_path: Path) -> Optional[Path]:
        """Find directory with hash-like name (40-char hex string)"""
        for item in workspace_path.iterdir():
            if not item.is_dir():
                continue
            if len(item.name) == HASH_DIR_LENGTH and all(
                c in "0123456789abcdef" for c in item.name.lower()
            ):
                if self._is_valid_sample_dir(item):
                    try:
                        mtime = item.stat().st_mtime
                        if time.time() - mtime < OUTPUT_SEARCH_WINDOW:
                            self.logger.info(f"Found hash-named directory: {item}")
                            return item
                    except (OSError, FileNotFoundError):
                        continue
        return None

    def _is_valid_sample_dir(self, dir_path: Path) -> bool:
        """Check if a directory is a valid sample directory"""
        required_files = ["model.py", "graph_net.json"]
        return all((dir_path / f).exists() for f in required_files)
