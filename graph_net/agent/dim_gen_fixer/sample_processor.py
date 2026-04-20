import os
import re

from graph_net.agent.dim_gen_fixer.claude_fix_agent import ClaudeFixAgent
from graph_net.agent.dim_gen_fixer.error_preprocessor import ErrorPreprocessor
from graph_net.agent.dim_gen_fixer.model_runner import ModelRunner
from graph_net.agent.dim_gen_fixer.fix_logger import FixLogger


class SampleProcessor:
    """Complete fix flow for a single sample (retry loop)."""

    def __init__(
        self,
        agent: ClaudeFixAgent,
        runner: ModelRunner,
        logger: FixLogger,
        max_retries: int,
        original_samples_dir: str,
        preprocessor: ErrorPreprocessor | None = None,
    ):
        self.agent = agent
        self.runner = runner
        self.logger = logger
        self.max_retries = max_retries
        self.original_samples_dir = original_samples_dir
        self.preprocessor = preprocessor

    def process(self, sample_path: str) -> dict:
        """
        Process a single sample, return result dict.

        Returns:
            {
                "sample_name": str,
                "sample_path": str,
                "status": str,          # OK(original)/OK(fixed)/Failed/Skipped(non-dim error)/Skipped(insufficient resources)
                "retry_count": int,
                "attempts": list,       # Records of each attempt
                "last_error_summary": str,
                "last_error_type": str,
                "skip_reason": str,
                "max_retries": int,
            }
        """
        sample_name = os.path.basename(sample_path)
        attempts = []
        retry_count = 0

        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_name}")
        print(f"Path: {sample_path}")
        print(f"{'='*60}")

        preprocess_action = None  # Track first preprocess action for reporting

        for retry in range(self.max_retries + 1):
            # Step B: Execute run_model
            print(f"\n  [{retry}] Running run_model...")
            result = self.runner.run(sample_path)

            if result["success"]:
                # Execution succeeded
                if retry == 0:
                    status = "OK(original)"
                    print("  OK: Sample passes without fix")
                else:
                    status = "OK(fixed)"
                    print(f"  OK: Sample passes after fix (attempt {retry})")

                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    status,
                    retry_count,
                    attempts,
                    "Execution succeeded",
                )
                return {
                    "sample_name": sample_name,
                    "sample_path": sample_path,
                    "status": status,
                    "retry_count": retry_count,
                    "attempts": attempts,
                    "last_error_summary": "",
                    "last_error_type": "",
                    "skip_reason": "",
                    "max_retries": self.max_retries,
                    "preprocess_action": preprocess_action,
                }

            # Execution failed
            error_output = result["output"]
            error_summary = self._extract_error_summary(error_output)
            error_type = self._extract_error_type(error_output)

            # Check for insufficient resources
            if self._is_resource_error(error_output):
                status = "Skipped(insufficient resources)"
                skip_reason = self._extract_resource_reason(error_output)
                print(f"  SKIP (insufficient resources): {skip_reason}")
                attempts.append(
                    {
                        "error_summary": error_summary,
                        "action": "skip_sample",
                        "reason": skip_reason,
                        "edits": [],
                    }
                )
                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    status,
                    retry_count,
                    attempts,
                    error_summary,
                )
                return {
                    "sample_name": sample_name,
                    "sample_path": sample_path,
                    "status": status,
                    "retry_count": retry_count,
                    "attempts": attempts,
                    "last_error_summary": error_summary,
                    "last_error_type": error_type,
                    "skip_reason": skip_reason,
                    "max_retries": self.max_retries,
                    "preprocess_action": preprocess_action,
                }

            if retry >= self.max_retries:
                break

            # Step A: Backup before first fix
            if retry == 0:
                self._backup_files(sample_path)

            # Step B: Try preprocessing before calling Claude API
            if self.preprocessor:
                preprocess_result = self.preprocessor.try_preprocess(
                    sample_path, error_output
                )

                if preprocess_result["action"] == "skip":
                    preprocess_action = "skip"
                    reason = preprocess_result.get(
                        "reason", "Auto-skipped by preprocessor"
                    )
                    status = "Skipped(non-dim error)"
                    print(f"  PREPROCESS SKIP: {reason}")
                    attempts.append(
                        {
                            "error_summary": error_summary,
                            "action": "skip_sample",
                            "reason": reason,
                            "edits": preprocess_result.get("edits", []),
                            "summary": preprocess_result.get("summary", ""),
                        }
                    )
                    self.logger.write_sample_log(
                        sample_name,
                        sample_path,
                        status,
                        retry_count,
                        attempts,
                        error_summary,
                    )
                    return {
                        "sample_name": sample_name,
                        "sample_path": sample_path,
                        "status": status,
                        "retry_count": retry_count,
                        "attempts": attempts,
                        "last_error_summary": error_summary,
                        "last_error_type": error_type,
                        "skip_reason": reason,
                        "max_retries": self.max_retries,
                        "preprocess_action": "skip",
                    }

                if preprocess_result["action"] == "fix":
                    if preprocess_action is None:
                        preprocess_action = "fix"
                    retry_count += 1
                    summary = preprocess_result.get("summary", "Preprocessed auto-fix")
                    print(f"  PREPROCESS FIX: {summary}")
                    attempts.append(
                        {
                            "error_summary": error_summary,
                            "action": "finish_fix",
                            "summary": summary,
                            "edits": preprocess_result.get("edits", []),
                        }
                    )
                    # Continue loop, re-execute run_model to verify
                    continue

            # Step C: Call Claude to analyze the error and fix
            print(f"  [{retry}] Calling Claude to analyze error...")
            fix_result = self.agent.fix_sample(
                sample_path=sample_path,
                error_output=error_output,
                original_samples_dir=self.original_samples_dir,
            )

            retry_count += 1

            if fix_result["action"] == "skip_sample":
                reason = fix_result.get("reason", "Unknown reason")
                # Determine if it's a dimension error that can't be fixed, or a non-dimension error
                if self._is_dim_error_but_unfixable(reason, error_output):
                    status = "Failed"
                else:
                    status = "Skipped(non-dim error)"
                print(f"  SKIP: {reason}")
                attempts.append(
                    {
                        "error_summary": error_summary,
                        "action": "skip_sample",
                        "reason": reason,
                        "edits": fix_result.get("edits", []),
                        "summary": fix_result.get("reason", ""),
                    }
                )
                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    status,
                    retry_count,
                    attempts,
                    error_summary,
                )
                return {
                    "sample_name": sample_name,
                    "sample_path": sample_path,
                    "status": status,
                    "retry_count": retry_count,
                    "attempts": attempts,
                    "last_error_summary": error_summary,
                    "last_error_type": error_type,
                    "skip_reason": reason,
                    "max_retries": self.max_retries,
                    "preprocess_action": preprocess_action,
                }

            # finish_fix - record the modification
            summary = fix_result.get("summary", "")
            print(f"  OK: Claude completed fix: {summary}")
            attempts.append(
                {
                    "error_summary": error_summary,
                    "action": "finish_fix",
                    "summary": summary,
                    "edits": fix_result.get("edits", []),
                }
            )
            # Continue loop, re-execute run_model

        # Reached max retries
        status = "Failed"
        print(f"  FAIL: Reached max retries ({self.max_retries})")
        self.logger.write_sample_log(
            sample_name,
            sample_path,
            status,
            retry_count,
            attempts,
            error_summary if attempts else "Reached max retries",
        )
        return {
            "sample_name": sample_name,
            "sample_path": sample_path,
            "status": status,
            "retry_count": retry_count,
            "attempts": attempts,
            "last_error_summary": error_summary if attempts else "",
            "last_error_type": error_type if attempts else "",
            "skip_reason": "",
            "max_retries": self.max_retries,
            "preprocess_action": preprocess_action,
        }

    @staticmethod
    def _backup_files(sample_path: str):
        """Backup model.py and weight_meta.py before the first fix."""
        for fname in ("model.py", "weight_meta.py"):
            src = os.path.join(sample_path, fname)
            bak = os.path.join(sample_path, fname + ".bak")
            if os.path.isfile(src) and not os.path.isfile(bak):
                try:
                    import shutil

                    shutil.copy2(src, bak)
                except Exception as e:
                    print(f"  Warning: Failed to backup {fname}: {e}")

    @staticmethod
    def _extract_error_summary(output: str) -> str:
        """Extract error output summary (last few lines)."""
        lines = output.strip().split("\n")
        # Take the last 5 lines
        summary_lines = lines[-5:] if len(lines) > 5 else lines
        return "\n".join(summary_lines)

    @staticmethod
    def _extract_error_type(output: str) -> str:
        """Extract error type from error output."""
        # Match common error types
        patterns = [
            r"(\w+Error):",
            r"(\w+Exception):",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                return matches[-1]
        return "Unknown"

    @staticmethod
    def _is_resource_error(output: str) -> bool:
        """Check if the error is due to insufficient resources."""
        resource_keywords = [
            "CUDA out of memory",
            "out of memory",
            "OOM",
            "TIMEOUT",
            "Killed",
        ]
        output_lower = output.lower()
        return any(kw.lower() in output_lower for kw in resource_keywords)

    @staticmethod
    def _extract_resource_reason(output: str) -> str:
        """Extract the reason for insufficient resources."""
        if "CUDA out of memory" in output or "out of memory" in output:
            return "CUDA out of memory"
        if "TIMEOUT" in output:
            return "Execution timeout"
        if "Killed" in output:
            return "Process killed (likely OOM)"
        return "Insufficient resources"

    @staticmethod
    def _is_dim_error_but_unfixable(reason: str, error_output: str) -> bool:
        """Check if it's a dimension error that cannot be fixed in-place.

        Only checks the skip reason from Claude, not the full error output,
        because the error output may contain dimension-related words in the
        traceback even for non-dimension errors (e.g., device mismatch).
        """
        dim_keywords = [
            "shape",
            "dimension",
            "size mismatch",
            "reshape",
            "view",
            "invalid for input of size",
            "only one dimension",
            "expand",
        ]
        reason_lower = reason.lower()
        # Only check the reason from Claude, trust Claude's classification
        return any(kw in reason_lower for kw in dim_keywords)
