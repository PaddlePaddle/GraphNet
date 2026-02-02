"""GraphNet Agent core implementation"""

import shutil
from pathlib import Path
from typing import Optional

from graph_net.hash_util import get_sha256_hash

from graph_net.agent.metadata_analyzer import ConfigMetadataAnalyzer
from graph_net.agent.code_generator import TemplateCodeGenerator
from graph_net.agent.graph_extractor import SubprocessGraphExtractor
from graph_net.agent.model_fetcher import HFFetcher
from graph_net.agent.utils.exceptions import (
    AnalysisError,
    CodeGenError,
    ExtractionError,
    VerificationError,
)
from graph_net.agent.utils.logger import setup_logger
from graph_net.agent.utils.workspace_manager import WorkspaceManager
from graph_net.agent.sample_verifier import BasicSampleVerifier


class GraphNetAgent:
    """GraphNet automatic sample extraction agent"""

    def __init__(
        self,
        workspace: str,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize GraphNet Agent

        Args:
            workspace: Workspace root directory
            hf_token: HuggingFace API token (optional)
        """
        self.workspace = WorkspaceManager(workspace)
        self.logger = setup_logger(
            "GraphNetAgent",
            log_file=self.workspace.get_log_path("agent"),
        )

        # Initialize modules
        self.model_fetcher = HFFetcher(
            cache_dir=str(self.workspace.models_dir),
            token=hf_token,
        )
        self.metadata_analyzer = ConfigMetadataAnalyzer()
        self.code_generator = TemplateCodeGenerator()
        self.graph_extractor = SubprocessGraphExtractor(
            workspace=str(self.workspace.workspace_root)
        )
        self.sample_verifier = BasicSampleVerifier()

    def extract_sample(self, model_id: str) -> bool:
        """
        Execute complete sample extraction pipeline from HuggingFace model ID

        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")

        Returns:
            True if sample extraction succeeded, False otherwise
        """
        try:
            self.logger.info(f"Starting extraction for model: {model_id}")

            model_dir = self._fetch_model(model_id)
            model_metadata = self._analyze_model(model_dir)
            script_path = self._generate_script(model_dir, model_metadata, model_id)
            sample_dir = self._extract_graph(script_path, model_id)

            self._generate_graph_hash(sample_dir)

            if self.is_duplicate_sample(sample_dir):
                self.logger.info("Duplicate sample detected, skipping verification")
                return True

            if not self.sample_verifier.verify(sample_dir):
                self.logger.error("Sample verification failed")
                return False

            self._archive_script(script_path, sample_dir)
            self.logger.info(f"Successfully extracted sample for {model_id}")
            return True

        except (AnalysisError, CodeGenError, ExtractionError, VerificationError) as e:
            self.logger.error(f"Extraction failed for {model_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error for {model_id}: {e}", exc_info=True)
            return False

    def _fetch_model(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub"""
        self.logger.info(f"Fetching model: {model_id}")
        model_dir = self.model_fetcher.download(model_id)
        self.logger.info(f"Model downloaded to: {model_dir}")
        return model_dir

    def _analyze_model(self, model_dir: Path):
        """Analyze model configuration to extract metadata"""
        self.logger.info("Analyzing model configuration")
        model_metadata = self.metadata_analyzer.analyze(model_dir)
        self.logger.info(
            f"Metadata: model_type={model_metadata.model_type}, vocab_size={model_metadata.vocab_size}"
        )
        return model_metadata

    def _generate_script(self, model_dir: Path, model_metadata, model_id: str) -> Path:
        """Generate run_model.py script based on metadata"""
        self.logger.info("Generating extraction script")
        generated_dir = self.workspace.get_generated_dir(model_id)
        script_path = self.code_generator.generate(
            model_dir, model_metadata, generated_dir
        )
        self.logger.info(f"Script generated: {script_path}")
        return script_path

    def _extract_graph(self, script_path: Path, model_id: str) -> Path:
        """Execute script to extract computation graph"""
        self.logger.info("Extracting computation graph")
        sample_dir = self.graph_extractor.extract(script_path, model_id)
        self.logger.info(f"Graph extracted to: {sample_dir}")
        return sample_dir

    def _archive_script(self, script_path: Path, sample_dir: Path) -> None:
        """Archive generated script to sample directory"""
        self.logger.info("Archiving extraction script")
        self.save_extraction_script(script_path, sample_dir)

    def _generate_graph_hash(self, sample_dir: Path) -> None:
        """Generate graph_hash.txt from model.py if it doesn't exist"""
        graph_hash_path = sample_dir / "graph_hash.txt"
        model_py_path = sample_dir / "model.py"

        if graph_hash_path.exists():
            return

        if not model_py_path.exists():
            self.logger.warning(f"model.py not found at {model_py_path}")
            return

        try:
            model_code = model_py_path.read_text()
            graph_hash = get_sha256_hash(model_code)
            graph_hash_path.write_text(graph_hash)
            self.logger.info(f"Generated graph_hash.txt: {graph_hash[:16]}...")
        except (OSError, IOError) as e:
            self.logger.warning(f"Failed to generate graph_hash.txt: {e}")

    def is_duplicate_sample(self, sample_dir: Path) -> bool:
        """Check if the extracted sample is a duplicate of an existing sample"""
        graph_hash_path = sample_dir / "graph_hash.txt"

        if not graph_hash_path.exists():
            return False

        try:
            current_hash = graph_hash_path.read_text().strip()
            samples_root = self.workspace.samples_dir

            if not samples_root.exists():
                return False

            for hash_file in samples_root.rglob("graph_hash.txt"):
                if hash_file == graph_hash_path:
                    continue
                try:
                    existing_hash = hash_file.read_text().strip()
                    if existing_hash == current_hash:
                        self.logger.info(f"Duplicate found: {hash_file.parent}")
                        return True
                except (OSError, IOError):
                    continue

            return False
        except (OSError, IOError) as e:
            self.logger.warning(f"Failed to check duplicate: {e}")
            return False

    def save_extraction_script(self, script_path: Path, sample_dir: Path) -> bool:
        """Save the generated extraction script to the sample directory"""
        try:
            target_path = sample_dir / "run_model.py"
            shutil.copy(script_path, target_path)
            self.logger.info(f"Script archived to: {target_path}")
            return True
        except (OSError, IOError, shutil.Error) as e:
            self.logger.error(f"Failed to archive script: {e}")
            return False
