"""GraphNet Agent core implementation"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from graph_net.hash_util import get_sha256_hash

from graph_net.agent.metadata_analyzer import ConfigMetadataAnalyzer
from graph_net.agent.code_generator import TemplateCodeGenerator
from graph_net.agent.code_generator.llm_code_fixer import LLMCodeFixer
from graph_net.agent.graph_extractor import SubprocessGraphExtractor
from graph_net.agent.model_fetcher import HFFetcher
from graph_net.agent.utils.error_classifier import GraphExtractionErrorClassifier
from graph_net.agent.utils.exceptions import (
    GraphExtractionErrorCategory,
    MetadataAnalysisError,
    CodeGenerationError,
    GraphExtractionError,
    SampleVerificationError,
)
from graph_net.agent.utils.logger import setup_logger
from graph_net.agent.utils.workspace_manager import WorkspaceManager
from graph_net.agent.sample_verifier import ForwardVerifier


class ExtractionStatus(str, Enum):
    """Extraction result status for a single model."""

    OK = "ok"
    VERIFY_FAILED = "verify_failed"
    EXTRACT_FAILED = "extract_failed"
    ERROR = "error"


class GraphNetAgent:
    """GraphNet automatic sample extraction agent"""

    def __init__(
        self,
        workspace: Optional[str] = None,
        hf_token: Optional[str] = None,
        llm_retry: bool = True,
        extract_timeout: Optional[int] = None,
        verify_timeout: Optional[int] = None,
        llm_timeout: int = 900,
    ):
        """
        Initialize GraphNet Agent

        Args:
            workspace:        Workspace root directory. Defaults to
                              $GRAPH_NET_EXTRACT_WORKSPACE or ~/graphnet_workspace.
            hf_token:         HuggingFace API token (optional)
            llm_retry:        If True and ducc/claude CLI is available, retry failed
                              extractions up to 2 times with LLM-fixed scripts.
            extract_timeout:  Timeout in seconds for graph extraction subprocess
                              (default None -> 1000s).
            verify_timeout:   Timeout in seconds for forward verification subprocess
                              (default None -> 300s).
            llm_timeout:      Timeout in seconds for LLM script fix (default: 600).
        """
        if workspace is None:
            workspace = os.environ.get(
                "GRAPH_NET_EXTRACT_WORKSPACE",
                os.path.expanduser("~/graphnet_workspace"),
            )
        self.workspace = WorkspaceManager(workspace)
        self.logger = setup_logger(
            "GraphNetAgent",
            log_file=self.workspace.get_log_path("agent"),
        )

        # Initialize modules
        self.model_fetcher = HFFetcher(
            cache_dir=str(self.workspace.models_dir),
            token=hf_token,
            max_retries=3,
            retry_delay=5,
        )
        self.metadata_analyzer = ConfigMetadataAnalyzer()
        self.code_generator = TemplateCodeGenerator()
        self.graph_extractor = SubprocessGraphExtractor(
            workspace=str(self.workspace.workspace_root),
            timeout=extract_timeout,
        )
        self.sample_verifier = ForwardVerifier(timeout=verify_timeout)

        # LLM fixer — only created when llm_retry is requested
        self.llm_fixer: Optional[LLMCodeFixer] = (
            LLMCodeFixer(timeout=llm_timeout) if llm_retry else None
        )

        # Track whether the last verify succeeded only because of timeout skip
        self.last_timeout_success = False

        # Error classifier for post-run reporting
        self.error_classifier = GraphExtractionErrorClassifier()

    def extract_sample(self, model_id: str) -> ExtractionStatus:
        """
        Execute complete sample extraction pipeline from HuggingFace model ID.

        On first failure the LLM fixer (ducc -p) is invoked up to 2 times to
        produce a repaired script. Each retry feeds the previous script and its
        error back to the LLM for further refinement.

        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")

        Returns:
            ExtractionStatus.OK              – extraction and verification both passed
            ExtractionStatus.VERIFY_FAILED   – extraction succeeded but verification failed
            ExtractionStatus.EXTRACT_FAILED  – extraction (or pre-extraction) failed
            ExtractionStatus.ERROR           – unexpected error
        """
        self.last_timeout_success = False
        try:
            self.logger.info(f"Starting extraction for model: {model_id}")

            model_dir = self._fetch_model(model_id)
            model_dir = self._resolve_model_dir(model_dir)
            model_metadata = self._analyze_model(model_dir)
            script_path = self._generate_script(model_dir, model_metadata, model_id)

            # ── First attempt (template script) ──────────────────────────
            try:
                sample_dir = self._extract_graph(script_path, model_id)
            except GraphExtractionError as first_err:
                if not self._is_llm_fixable_error(first_err):
                    self.logger.warning(
                        f"Extraction error is not fixable by LLM, skipping retry: {first_err}"
                    )
                    raise first_err
                sample_dir = self._llm_retry(
                    first_err, script_path, model_dir, model_id
                )

            self._generate_graph_hash(sample_dir)
            self._fix_model_name(sample_dir, model_id)

            if self.is_duplicate_sample(sample_dir):
                self.logger.info("Duplicate sample detected, skipping verification")
                return ExtractionStatus.OK

            if not self.sample_verifier.verify(sample_dir):
                self.logger.error("Sample verification failed")
                self.error_classifier.classify_and_record(
                    model_id,
                    Exception("Sample verification failed"),
                )
                return ExtractionStatus.VERIFY_FAILED

            if getattr(self.sample_verifier, "last_timeout_success", False):
                self.last_timeout_success = True
                self.logger.info(
                    f"Sample verification for {model_id} passed via timeout skip"
                )

            self.logger.info(f"Successfully extracted sample for {model_id}")
            return ExtractionStatus.OK

        except SampleVerificationError as e:
            self.logger.error(f"Extraction failed for {model_id}: {e}")
            self.error_classifier.classify_and_record(model_id, e)
            return ExtractionStatus.VERIFY_FAILED
        except (MetadataAnalysisError, CodeGenerationError, GraphExtractionError) as e:
            self.logger.error(f"Extraction failed for {model_id}: {e}")
            self.error_classifier.classify_and_record(model_id, e)
            return ExtractionStatus.EXTRACT_FAILED
        except Exception as e:
            self.logger.error(f"Unexpected error for {model_id}: {e}", exc_info=True)
            self.error_classifier.classify_and_record(model_id, e)
            return ExtractionStatus.ERROR

    @staticmethod
    def _is_llm_fixable_error(err: GraphExtractionError) -> bool:
        """Decide whether an extraction error is worth retrying with LLM.

        Only allow LLM retry for script logic errors (non-zero return code).
        All other categories (timeout, infrastructure, missing model, etc.)
        are not fixable by rewriting the script.
        """
        category = GraphExtractionErrorClassifier.classify_from_exception(err)
        return category == GraphExtractionErrorCategory.SCRIPT_EXECUTION_FAILED

    def _llm_retry(
        self,
        first_err: GraphExtractionError,
        script_path: Path,
        model_dir: Path,
        model_id: str,
    ) -> tuple[Path, Path]:
        """
        On extraction failure: ask the LLM to fix the script and retry, up to 2 times.
        Each attempt feeds the previous script + its error back to the LLM.

        Returns:
            (sample_dir, successful_script_path)

        Raises GraphExtractionError if LLM fix is unavailable or both attempts fail.
        """
        if self.llm_fixer is None or not self.llm_fixer.available:
            self.logger.warning(
                "LLM retry disabled or ducc not available; re-raising original error."
            )
            raise first_err

        generated_dir = self.workspace.get_generated_dir(model_id)
        err = first_err
        current_script = script_path

        for attempt in range(1, 3):  # attempt 1, 2
            self.logger.warning(
                f"Extraction failed (attempt {attempt}/2): {err}\n"
                f"Invoking LLM to fix the script..."
            )
            fixed_path = self.llm_fixer.fix(
                script_path=current_script,
                error_msg=str(err),
                model_dir=model_dir,
                model_id=model_id,
                output_dir=generated_dir,
                attempt=attempt,
            )
            self.logger.info(f"Retrying extraction with LLM-fixed script: {fixed_path}")
            try:
                sample_dir = self._extract_graph(fixed_path, model_id)
                return sample_dir
            except GraphExtractionError as retry_err:
                err = retry_err
                current_script = fixed_path  # 第二次把上一次修复的脚本+新报错再喂给 LLM

        raise err

    def _fetch_model(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub"""
        self.logger.info(f"Fetching model: {model_id}")
        model_dir = self.model_fetcher.download(model_id)
        self.logger.info(f"Model downloaded to: {model_dir}")
        return model_dir

    def _resolve_model_dir(self, model_dir: Path) -> Path:
        """
        For diffusers pipeline repos (identified by model_index.json at root),
        resolve to the UNet subdirectory which contains the actual UNet config.
        Returns model_dir unchanged for non-pipeline repos.
        """
        model_index = model_dir / "model_index.json"
        if not model_index.exists():
            return model_dir

        # It's a diffusers pipeline — find the unet subdirectory
        unet_dir = model_dir / "unet"
        if unet_dir.is_dir() and (unet_dir / "config.json").exists():
            self.logger.info(
                f"Detected diffusers pipeline; using UNet subdir: {unet_dir}"
            )
            return unet_dir

        # Pipeline without unet/ (e.g., image-to-image or non-SD pipeline)
        self.logger.warning(
            f"Diffusers pipeline detected but no unet/ subdir found in {model_dir}; "
            "proceeding with root dir."
        )
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
        # Override model_id in metadata with the original HF model_id so that
        # extract(name=...) uses the short model name, not a snapshot hash.
        model_metadata.model_id = model_id
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

    def _fix_model_name(self, sample_dir: Path, model_id: str) -> None:
        """将 graph_net.json 中的 model_name 修正为原始 HuggingFace model_id（org/model）"""
        for json_path in [
            sample_dir / "graph_net.json",
            *sample_dir.glob("subgraph_*/graph_net.json"),
        ]:
            if not json_path.exists():
                continue
            try:
                data = json.loads(json_path.read_text())
                if data.get("model_name") != model_id:
                    data["model_name"] = model_id
                    json_path.write_text(json.dumps(data, indent=4))
            except (OSError, json.JSONDecodeError) as e:
                self.logger.warning(f"Failed to fix model_name in {json_path}: {e}")

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
