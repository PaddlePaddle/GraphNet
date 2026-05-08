"""GraphNet Agent core implementation"""

import json
import shutil
from pathlib import Path
from typing import Optional

from graph_net.hash_util import get_sha256_hash

from graph_net.agent.metadata_analyzer import ConfigMetadataAnalyzer
from graph_net.agent.code_generator import TemplateCodeGenerator
from graph_net.agent.code_generator.llm_code_fixer import LLMCodeFixer
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
from graph_net.agent.sample_verifier import ForwardVerifier


class GraphNetAgent:
    """GraphNet automatic sample extraction agent"""

    def __init__(
        self,
        workspace: str = "/home/luotao02/workspace",
        hf_token: Optional[str] = None,
        llm_retry: bool = True,
        max_model_size_b: float = 20.0,
        timeout: int = 1000,
    ):
        """
        Initialize GraphNet Agent

        Args:
            workspace:        Workspace root directory
            hf_token:         HuggingFace API token (optional)
            llm_retry:        If True and ducc/claude CLI is available, retry failed
                              extractions up to 2 times with LLM-fixed scripts.
            max_model_size_b: Maximum model size in billions of parameters to attempt.
                              Models exceeding this limit are skipped (AnalysisError).
                              Set to (total_RAM_gb × 0.7 / num_workers / 4) for safe
                              concurrent extraction.  Default: 20.0B.
            timeout:          Subprocess execution timeout in seconds. Default: 1000s.
        """
        self.workspace = WorkspaceManager(workspace)
        self.max_model_size_b = max_model_size_b
        self.logger = setup_logger(
            name="graph_net_agent",
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
            timeout=timeout,
        )
        self.sample_verifier = ForwardVerifier()

        # LLM fixer — only created when llm_retry is requested
        self.llm_fixer: Optional[LLMCodeFixer] = LLMCodeFixer() if llm_retry else None

    def extract_sample(self, model_id: str) -> bool:
        """
        Execute complete sample extraction pipeline from HuggingFace model ID.

        On first failure the LLM fixer (ducc -p) is invoked up to 2 times to
        produce a repaired script. Each retry feeds the previous script and its
        error back to the LLM for further refinement.

        Args:
            model_id: HuggingFace model ID (e.g., "bert-base-uncased")

        Returns:
            True if sample extraction succeeded, False otherwise
        """
        sample_dir: Optional[Path] = None
        try:
            self.logger.info(f"Starting extraction for model: {model_id}")

            model_dir = self._fetch_model(model_id)
            model_metadata = self._analyze_model(model_dir, self.max_model_size_b)
            script_path = self._generate_script(model_dir, model_metadata, model_id)

            # ── First attempt (template script) ──────────────────────────
            try:
                sample_dir = self._extract_graph(script_path, model_id)
            except ExtractionError as first_err:
                sample_dir = self._llm_retry(
                    first_err, script_path, model_dir, model_id
                )

            self._generate_graph_hash(sample_dir)
            self._fix_model_name(sample_dir, model_id)

            if self.is_duplicate_sample(sample_dir):
                self.logger.info("Duplicate sample detected, skipping verification")
                self._move_sample(sample_dir, self.workspace.success_dir)
                return True

            if not self.sample_verifier.verify(sample_dir):
                self.logger.error("Sample verification failed")
                self._move_sample(sample_dir, self.workspace.failed_dir)
                return False

            self._move_sample(sample_dir, self.workspace.success_dir)
            self.logger.info(f"Successfully extracted sample for {model_id}")
            return True

        except (AnalysisError, CodeGenError, ExtractionError, VerificationError) as e:
            self.logger.error(f"Extraction failed for {model_id}: {e}")
            if sample_dir and sample_dir.exists():
                self._move_sample(sample_dir, self.workspace.failed_dir)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error for {model_id}: {e}", exc_info=True)
            if sample_dir and sample_dir.exists():
                self._move_sample(sample_dir, self.workspace.failed_dir)
            return False

    def _llm_retry(
        self,
        first_err: ExtractionError,
        script_path: Path,
        model_dir: Path,
        model_id: str,
    ) -> tuple[Path, Path]:
        """
        On extraction failure: ask the LLM to fix the script and retry, up to 2 times.
        Each attempt feeds the previous script + its error back to the LLM.

        Returns:
            (sample_dir, successful_script_path)

        Raises ExtractionError if LLM fix is unavailable or both attempts fail.
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
            except ExtractionError as retry_err:
                err = retry_err
                current_script = fixed_path  # 第二次把上一次修复的脚本+新报错再喂给 LLM

        raise err

    def _fetch_model(self, model_id: str) -> Path:
        """Download model from HuggingFace Hub"""
        self.logger.info(f"Fetching model: {model_id}")
        model_dir = self.model_fetcher.download(model_id)
        self.logger.info(f"Model downloaded to: {model_dir}")
        return model_dir

    def _analyze_model(self, model_dir: Path, max_param_b: float = 20.0):
        """Analyze model configuration to extract metadata"""
        self.logger.info("Analyzing model configuration")
        model_metadata = self.metadata_analyzer.analyze(model_dir, max_param_b=max_param_b)
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
        """Generate graph_hash.txt from model.py.

        - Single-graph: hash root model.py → sample_dir/graph_hash.txt
        - Multi-subgraph: hash each subgraph_N/model.py → subgraph_N/graph_hash.txt
          (no top-level graph_hash.txt for multi-subgraph models)
        """
        root_model = sample_dir / "model.py"
        if root_model.exists():
            # Single-graph model
            graph_hash_path = sample_dir / "graph_hash.txt"
            if graph_hash_path.exists():
                return
            try:
                graph_hash = get_sha256_hash(root_model.read_text())
                graph_hash_path.write_text(graph_hash)
                self.logger.info(f"Generated graph_hash.txt: {graph_hash[:16]}...")
            except (OSError, IOError) as e:
                self.logger.warning(f"Failed to generate graph_hash.txt: {e}")
        else:
            # Multi-subgraph model: generate per-subgraph hashes
            subgraph_dirs = sorted(sample_dir.glob("subgraph_*"))
            if not subgraph_dirs:
                self.logger.warning(f"No model.py or subgraph dirs found in {sample_dir}")
                return
            for subgraph_dir in subgraph_dirs:
                model_py = subgraph_dir / "model.py"
                hash_path = subgraph_dir / "graph_hash.txt"
                if hash_path.exists() or not model_py.exists():
                    continue
                try:
                    graph_hash = get_sha256_hash(model_py.read_text())
                    hash_path.write_text(graph_hash)
                    self.logger.info(
                        f"Generated {subgraph_dir.name}/graph_hash.txt: {graph_hash[:16]}..."
                    )
                except (OSError, IOError) as e:
                    self.logger.warning(
                        f"Failed to generate graph_hash.txt for {subgraph_dir}: {e}"
                    )

    def _move_sample(self, sample_dir: Path, dest_parent: Path) -> Path:
        """Move sample_dir into dest_parent/, overwriting if destination exists"""
        dest = dest_parent / sample_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(sample_dir), str(dest))
        self.logger.info(f"Moved sample to: {dest}")
        return dest

    def is_duplicate_sample(self, sample_dir: Path) -> bool:
        """Check if the extracted sample is a duplicate of an existing sample.

        - Single-graph: compare root graph_hash.txt against existing root hashes.
        - Multi-subgraph: compare frozenset of all subgraph_*/graph_hash.txt hashes.
        """
        try:
            root_model = sample_dir / "model.py"
            if root_model.exists():
                # Single-graph
                graph_hash_path = sample_dir / "graph_hash.txt"
                if not graph_hash_path.exists():
                    return False
                current_hash = graph_hash_path.read_text().strip()
                for search_root in [self.workspace.success_dir, self.workspace.samples_dir]:
                    if not search_root.exists():
                        continue
                    for existing_dir in search_root.iterdir():
                        if not existing_dir.is_dir() or existing_dir == sample_dir:
                            continue
                        hash_file = existing_dir / "graph_hash.txt"
                        if not hash_file.exists():
                            continue
                        try:
                            if hash_file.read_text().strip() == current_hash:
                                self.logger.info(f"Duplicate found: {existing_dir}")
                                return True
                        except (OSError, IOError):
                            continue
            else:
                # Multi-subgraph: compare the set of subgraph hashes
                current_hashes = frozenset(
                    h.read_text().strip()
                    for h in sample_dir.glob("subgraph_*/graph_hash.txt")
                )
                if not current_hashes:
                    return False
                for search_root in [self.workspace.success_dir, self.workspace.samples_dir]:
                    if not search_root.exists():
                        continue
                    for existing_dir in search_root.iterdir():
                        if not existing_dir.is_dir() or existing_dir == sample_dir:
                            continue
                        existing_hashes = frozenset(
                            h.read_text().strip()
                            for h in existing_dir.glob("subgraph_*/graph_hash.txt")
                        )
                        if existing_hashes and existing_hashes == current_hashes:
                            self.logger.info(f"Duplicate found: {existing_dir}")
                            return True
        except (OSError, IOError) as e:
            self.logger.warning(f"Failed to check duplicate: {e}")
        return False
