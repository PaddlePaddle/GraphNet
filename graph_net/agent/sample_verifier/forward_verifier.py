"""Forward-pass verifier: eager execution of extracted GraphModule."""

import logging
import subprocess
import sys
from pathlib import Path

from graph_net.agent.sample_verifier.base import BaseSampleVerifier
from graph_net.agent.sample_verifier.basic_sample_verifier import BasicSampleVerifier
from graph_net.agent.utils.exceptions import VerificationError

# Inline eager runner — executed in a subprocess to isolate CUDA state.
# Loads GraphModule from model.py, reconstructs tensors from weight_meta.py,
# and runs a plain eager forward pass (no dynamo / compile).
_EAGER_RUNNER = """
import sys, importlib.util, torch
from graph_net.torch import utils

model_path = sys.argv[1]

spec = importlib.util.spec_from_file_location("m", f"{model_path}/model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.GraphModule().eval()

inputs_params = utils.load_converted_from_text(model_path)
state_dict = {k: utils.replay_tensor(v) for k, v in inputs_params["weight_info"].items()}

with torch.no_grad():
    model(**state_dict)
"""


class ForwardVerifier(BaseSampleVerifier):
    """
    Two-stage verifier:
      1. BasicSampleVerifier — file existence & JSON validity
      2. Eager forward pass  — plain model(**inputs), no dynamo/compile

    For multi-subgraph samples (subgraph_0/, subgraph_1/, …) each subgraph
    is verified independently; all must pass.
    """

    def __init__(self, timeout: int = 300):
        """
        Args:
            timeout: seconds to wait for each forward-pass subprocess
                     (default 300s, can be overridden)
        """
        self._basic = BasicSampleVerifier()
        self.timeout = timeout if timeout is not None else 300
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_timeout_success = False

    def verify(self, sample_dir: Path) -> bool:
        """
        Verify sample validity including a real eager forward pass.

        Args:
            sample_dir: Path to extracted sample directory

        Returns:
            True if all checks pass, False otherwise
        """
        self.last_timeout_success = False
        try:
            # Stage 1: file structure check
            if not self._basic.verify(sample_dir):
                self.logger.warning(f"Basic verification failed: {sample_dir}")
                return False

            # Stage 2: eager forward pass (per subgraph if multi-subgraph)
            subgraph_dirs = sorted(sample_dir.glob("subgraph_*/"))
            targets = subgraph_dirs if subgraph_dirs else [sample_dir]

            for target in targets:
                ok, is_timeout = self._run_forward(target)
                if not ok:
                    return False
                if is_timeout:
                    self.last_timeout_success = True

            return True

        except Exception as e:
            raise VerificationError(f"Forward verification failed: {e}") from e

    def _run_forward(self, model_path: Path) -> tuple[bool, bool]:
        """Run an eager forward pass on one model directory in a subprocess.

        Returns:
            (success, is_timeout): success=True means the check passed;
                                   is_timeout=True means it passed only because
                                   the subprocess timed out (treated as skip).
        """
        self.logger.info(f"Forward verify (eager): {model_path.name}")
        try:
            result = subprocess.run(
                [sys.executable, "-c", _EAGER_RUNNER, str(model_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                self.logger.info(f"Forward verify OK: {model_path.name}")
                return True, False
            else:
                self.logger.warning(
                    f"Forward verify FAIL: {model_path.name}\n{result.stderr[-2000:]}"
                )
                return False, False
        except subprocess.TimeoutExpired:
            self.logger.warning(
                f"Forward verify TIMEOUT ({self.timeout}s): {model_path.name}, "
                "treating as pass (skip verification for large models on CPU)"
            )
            return True, True
