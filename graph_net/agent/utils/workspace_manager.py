"""Workspace management utilities"""

from pathlib import Path
from typing import Optional


class WorkspaceManager:
    """Manages Agent workspace directory structure"""

    def __init__(self, workspace_root: str):
        """
        Args:
            workspace_root: Root directory for workspace
        """
        self.workspace_root = Path(workspace_root)
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary workspace directories"""
        dirs = [
            self.models_dir,
            self.generated_dir,
            self.samples_dir,
            self.logs_dir,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def models_dir(self) -> Path:
        """Directory for downloaded models"""
        return self.workspace_root / "models"

    @property
    def generated_dir(self) -> Path:
        """Directory for generated scripts"""
        return self.workspace_root / "generated"

    @property
    def samples_dir(self) -> Path:
        """Directory for extracted samples"""
        return self.workspace_root / "samples"

    @property
    def logs_dir(self) -> Path:
        """Directory for logs"""
        return self.workspace_root / "logs"

    def get_model_dir(self, model_id: str) -> Path:
        """Get directory path for a specific model"""
        return self.models_dir / model_id.replace("/", "_")

    def get_generated_dir(self, model_id: str) -> Path:
        """Get directory path for generated script"""
        return self.generated_dir / model_id.replace("/", "_")

    def get_sample_dir(self, model_id: str) -> Path:
        """Get directory path for extracted sample"""
        return self.samples_dir / model_id.replace("/", "_")

    def get_log_path(self, model_id: str, timestamp: Optional[str] = None) -> Path:
        """Get log file path"""
        if timestamp is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_id = model_id.replace("/", "_")
        return self.logs_dir / f"{safe_model_id}_{timestamp}.log"
