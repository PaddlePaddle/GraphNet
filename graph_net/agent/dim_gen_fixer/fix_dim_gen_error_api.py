"""
fix_dim_gen_error_api.py - Automatically fix dimension errors in dimension-generalized
samples using the Anthropic API (Claude).

Implements intelligent error analysis and repair through Anthropic Messages API + Tool Use.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python graph_net/agent/dim_gen_fixer/fix_dim_gen_error_api.py --samples-dir /path/to/samples --max-samples 1 --max-retries 2
"""

import argparse
import os
import sys
from datetime import datetime

from graph_net.agent.dim_gen_fixer.claude_fix_agent import ClaudeFixAgent
from graph_net.agent.dim_gen_fixer.error_preprocessor import ErrorPreprocessor
from graph_net.agent.dim_gen_fixer.model_runner import ModelRunner
from graph_net.agent.dim_gen_fixer.fix_logger import FixLogger
from graph_net.agent.dim_gen_fixer.sample_processor import SampleProcessor


# Force unbuffered stdout for background process logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class MainOrchestrator:
    """Main workflow: scan -> process samples -> generate summary report.

    Parallelism strategy:
    - run_model execution is parallelized across GPUs (via ThreadPoolExecutor)
    - Claude API calls are serialized (single API key cannot handle concurrent
      requests at the gateway, concurrent calls get queued/blocked)
    - Flow: batch-run_model on all GPUs -> collect failures -> serial API fix
      -> batch-run_model to verify -> repeat
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(
            args.output_dir, f"fix_sample_logs-{self.timestamp}"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        self.agent = ClaudeFixAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            custom_header=args.custom_header,
        )
        self.logger = FixLogger(log_dir=self.log_dir)

        # Create error preprocessor for rule-based handling
        self.preprocessor = ErrorPreprocessor(
            original_samples_dir=args.original_samples_dir
        )

        # Parse GPU list for parallel run_model execution
        self.gpu_ids = self._parse_gpus(args.gpus)

        # ModelRunner for the processor: use first GPU if available,
        # otherwise no GPU binding (sequential mode)
        default_gpu = self.gpu_ids[0] if self.gpu_ids else None
        self.processor = SampleProcessor(
            agent=self.agent,
            runner=ModelRunner(timeout=args.timeout, gpu_id=default_gpu),
            logger=self.logger,
            max_retries=args.max_retries,
            original_samples_dir=args.original_samples_dir,
            preprocessor=self.preprocessor,
        )

    @staticmethod
    def _parse_gpus(gpus_str: str) -> list:
        """Parse GPU list from comma-separated string like '0,1,2,3'."""
        if not gpus_str:
            return []
        try:
            return [int(g.strip()) for g in gpus_str.split(",") if g.strip()]
        except ValueError:
            print(
                f"Error: Invalid --gpus format: {gpus_str}. "
                "Expected comma-separated integers like '0,1,2,3'"
            )
            sys.exit(1)

    def run(self):
        """Execute the main workflow."""
        print(f"Log directory: {self.log_dir}")
        print(f"Max retries: {self.args.max_retries}")
        print(f"Timeout: {self.args.timeout}s")
        print(f"Claude model: {self.args.model}")
        print(f"Original samples directory: {self.args.original_samples_dir}")
        print(f"GPUs for parallel run_model: {self.gpu_ids or 'sequential'}")

        # Step 3.1: Scan samples
        samples = self._scan_samples()
        print(f"\nFound {len(samples)} samples to process")

        if not samples:
            print("No samples to process, exiting.")
            return

        # If no sample_list_with_status provided, generate one in the log directory
        self.sample_list_path = self.args.sample_list_with_status
        if not self.sample_list_path:
            self.sample_list_path = os.path.join(self.log_dir, "sample_list.txt")
            self._generate_sample_list(self.sample_list_path, samples)
            print(f"Generated sample list: {self.sample_list_path}")

        # Step 3.2: Process samples
        if self.gpu_ids:
            results = self._process_samples_parallel(samples)
        else:
            results = self._process_samples_sequential(samples)

        # Step 3.3: Generate summary report
        samples_dir = self.args.samples_dir or ""
        self.logger.write_report(samples_dir, results)

        # Print summary
        self._print_summary(results)

    def _process_samples_sequential(self, samples: list) -> list:
        """Process samples sequentially (no GPU parallelism)."""
        results = []
        for i, sample_path in enumerate(samples, 1):
            print(f"\n--- [{i}/{len(samples)}] ---")
            result = self.processor.process(sample_path)
            results.append(result)
            self._update_sample_status(
                self.sample_list_path, sample_path, result["status"]
            )
        return results

    def _process_samples_parallel(self, samples: list) -> list:
        """Process samples with parallel run_model, serial API calls.

        Strategy:
        1. Batch-run all samples' run_model in parallel across GPUs
        2. Samples that pass -> record as OK(original)
        3. Samples that fail -> process through serial fix loop
           (run_model runs on the next available GPU, API calls are serial)
        """

        # Phase 1: Parallel initial run_model to identify failures
        print(f"\n=== Phase 1: Parallel run_model ({len(self.gpu_ids)} GPUs) ===")
        phase1_results = []  # Full result dicts for handled samples
        failed_samples = []  # (sample_path, error_output) for samples that need fixing

        run_results = self._batch_run_model(samples)

        for sample_path, run_result in zip(samples, run_results):
            sample_name = os.path.basename(sample_path)
            if run_result["success"]:
                print(f"  PASS: {sample_name}")
                self.logger.write_sample_log(
                    sample_name,
                    sample_path,
                    "OK(original)",
                    0,
                    [],
                    "Execution succeeded",
                )
                self._update_sample_status(
                    self.sample_list_path, sample_path, "OK(original)"
                )
                phase1_results.append(
                    {
                        "sample_name": sample_name,
                        "sample_path": sample_path,
                        "status": "OK(original)",
                        "retry_count": 0,
                        "attempts": [],
                        "last_error_summary": "",
                        "last_error_type": "",
                        "skip_reason": "",
                        "max_retries": self.args.max_retries,
                        "preprocess_action": None,
                    }
                )
            else:
                error_output = run_result["output"]
                error_summary = SampleProcessor._extract_error_summary(error_output)
                error_type = SampleProcessor._extract_error_type(error_output)

                # Check resource errors
                if SampleProcessor._is_resource_error(error_output):
                    skip_reason = SampleProcessor._extract_resource_reason(error_output)
                    print(f"  SKIP (resources): {sample_name} - {skip_reason}")
                    self.logger.write_sample_log(
                        sample_name,
                        sample_path,
                        "Skipped(insufficient resources)",
                        0,
                        [
                            {
                                "error_summary": error_summary,
                                "action": "skip_sample",
                                "reason": skip_reason,
                                "edits": [],
                            }
                        ],
                        error_summary,
                    )
                    self._update_sample_status(
                        self.sample_list_path,
                        sample_path,
                        "Skipped(insufficient resources)",
                    )
                    phase1_results.append(
                        {
                            "sample_name": sample_name,
                            "sample_path": sample_path,
                            "status": "Skipped(insufficient resources)",
                            "retry_count": 0,
                            "attempts": [
                                {
                                    "error_summary": error_summary,
                                    "action": "skip_sample",
                                    "reason": skip_reason,
                                    "edits": [],
                                }
                            ],
                            "last_error_summary": error_summary,
                            "last_error_type": error_type,
                            "skip_reason": skip_reason,
                            "max_retries": self.args.max_retries,
                            "preprocess_action": None,
                        }
                    )
                elif self.preprocessor:
                    # Try preprocessing to skip non-dim errors
                    preprocess_result = self.preprocessor.try_preprocess(
                        sample_path, error_output
                    )
                    if preprocess_result["action"] == "skip":
                        reason = preprocess_result.get("reason", "Auto-skipped")
                        print(f"  SKIP (preprocess): {sample_name} - {reason[:100]}")
                        self.logger.write_sample_log(
                            sample_name,
                            sample_path,
                            "Skipped(non-dim error)",
                            0,
                            [
                                {
                                    "error_summary": error_summary,
                                    "action": "skip_sample",
                                    "reason": reason,
                                    "edits": [],
                                    "summary": preprocess_result.get("summary", ""),
                                }
                            ],
                            error_summary,
                        )
                        self._update_sample_status(
                            self.sample_list_path, sample_path, "Skipped(non-dim error)"
                        )
                        phase1_results.append(
                            {
                                "sample_name": sample_name,
                                "sample_path": sample_path,
                                "status": "Skipped(non-dim error)",
                                "retry_count": 0,
                                "attempts": [
                                    {
                                        "error_summary": error_summary,
                                        "action": "skip_sample",
                                        "reason": reason,
                                        "edits": [],
                                        "summary": preprocess_result.get("summary", ""),
                                    }
                                ],
                                "last_error_summary": error_summary,
                                "last_error_type": error_type,
                                "skip_reason": reason,
                                "max_retries": self.args.max_retries,
                                "preprocess_action": "skip",
                            }
                        )
                    else:
                        print(f"  FAIL: {sample_name}")
                        failed_samples.append((sample_path, error_output))
                else:
                    print(f"  FAIL: {sample_name}")
                    failed_samples.append((sample_path, error_output))

        print(
            f"\nPhase 1 done: {len(phase1_results)} handled, "
            f"{len(failed_samples)} failed, need fix"
        )

        # Phase 2: Serial fix loop for failed samples
        # run_model is dispatched to GPUs in round-robin, API calls are serial
        results = list(phase1_results)

        # Process failed samples one by one (serial API, GPU-parallel run_model)
        if failed_samples:
            print(
                f"\n=== Phase 2: Fix {len(failed_samples)} failed samples (serial API) ==="
            )
        for i, (sample_path, _) in enumerate(failed_samples, 1):
            print(f"\n--- Fix [{i}/{len(failed_samples)}] ---")
            result = self.processor.process(sample_path)
            results.append(result)
            self._update_sample_status(
                self.sample_list_path, sample_path, result["status"]
            )

        # Sort by original sample order
        sample_order = {s: idx for idx, s in enumerate(samples)}
        results.sort(key=lambda r: sample_order.get(r.get("sample_path", ""), 0))
        return results

    def _batch_run_model(self, samples: list) -> list:
        """Run model on all samples in parallel across GPUs. Returns results list."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        num_gpus = len(self.gpu_ids)
        # Assign samples to GPUs round-robin
        gpu_assignments = {}
        for i, sample_path in enumerate(samples):
            gpu_id = self.gpu_ids[i % num_gpus]
            gpu_assignments.setdefault(gpu_id, []).append(sample_path)

        # Collect all (sample_path, gpu_id) pairs in order
        tasks = []
        for i, sample_path in enumerate(samples):
            tasks.append((sample_path, self.gpu_ids[i % num_gpus]))

        results_map = {}

        def run_on_gpu(sample_path: str, gpu_id: int) -> tuple:
            runner = ModelRunner(timeout=self.args.timeout, gpu_id=gpu_id)
            return (sample_path, runner.run(sample_path))

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {executor.submit(run_on_gpu, sp, gid): sp for sp, gid in tasks}
            for future in as_completed(futures):
                try:
                    sample_path, run_result = future.result()
                    results_map[sample_path] = run_result
                except Exception as e:
                    sample_path = futures[future]
                    results_map[sample_path] = {
                        "success": False,
                        "returncode": -1,
                        "output": f"ERROR: {e}",
                    }

        # Return results in original sample order
        return [results_map[sp] for sp in samples]

    def _generate_sample_list(self, list_path: str, samples: list):
        """Generate a sample_list.txt file with all samples marked as FAILED."""
        with open(list_path, "w", encoding="utf-8") as f:
            for sample_path in samples:
                f.write(f"FAILED\t{sample_path}\n")

    def _scan_samples(self) -> list:
        """Scan samples to process, return list of paths."""
        if self.args.sample_list_with_status:
            return self._scan_from_status_file(self.args.sample_list_with_status)
        elif self.args.samples_dir:
            return self._scan_from_dir(self.args.samples_dir)
        else:
            print("Error: Must provide --samples-dir or --sample-list-with-status")
            sys.exit(1)

    def _scan_from_status_file(self, status_file: str) -> list:
        """Scan from a status file with sample list."""
        if not os.path.isfile(status_file):
            print(f"Error: Sample list file not found: {status_file}")
            sys.exit(1)

        samples = []
        with open(status_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                status = parts[0].strip()
                path = parts[1].strip()
                # FAILED or blank status -> needs processing
                if status in ("", "FAILED"):
                    samples.append(path)

        if self.args.max_samples > 0:
            samples = samples[: self.args.max_samples]

        return samples

    def _scan_from_dir(self, samples_dir: str) -> list:
        """Scan all subdirectories containing model.py from the given directory."""
        if not os.path.isdir(samples_dir):
            print(f"Error: Sample directory not found: {samples_dir}")
            sys.exit(1)

        samples = []
        for root, dirs, files in os.walk(samples_dir):
            if "model.py" in files:
                samples.append(root)

        samples.sort()

        if self.args.max_samples > 0:
            samples = samples[: self.args.max_samples]

        return samples

    @staticmethod
    def _update_sample_status(status_file: str, sample_path: str, status: str):
        """Update sample status in-place in the sample_list_with_status file."""
        if status.startswith("OK"):
            new_status = "PASS"
        else:
            new_status = "FAILED"

        try:
            with open(status_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            updated = []
            for line in lines:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) >= 2 and parts[1].strip() == sample_path:
                    updated.append(f"{new_status}\t{sample_path}\n")
                else:
                    updated.append(line)

            with open(status_file, "w", encoding="utf-8") as f:
                f.writelines(updated)
        except Exception as e:
            print(f"  Warning: Failed to update status file: {e}")

    @staticmethod
    def _print_summary(results: list):
        """Print a brief summary."""
        total = len(results)
        original_ok = sum(1 for r in results if r["status"] == "OK(original)")
        fixed_ok = sum(1 for r in results if r["status"] == "OK(fixed)")
        failed = sum(1 for r in results if r["status"] == "Failed")
        skipped_dim = sum(1 for r in results if r["status"] == "Skipped(non-dim error)")
        skipped_res = sum(
            1 for r in results if r["status"] == "Skipped(insufficient resources)"
        )

        # Preprocessing statistics
        preprocessed_skip = sum(
            1 for r in results if r.get("preprocess_action") == "skip"
        )
        preprocessed_fix_ok = sum(
            1
            for r in results
            if r.get("preprocess_action") == "fix" and r["status"].startswith("OK")
        )
        preprocessed_fix_failed = sum(
            1
            for r in results
            if r.get("preprocess_action") == "fix" and r["status"] == "Failed"
        )
        api_fix_ok = sum(
            1
            for r in results
            if r.get("preprocess_action") in ("unhandled", None)
            and r["status"] == "OK(fixed)"
        )

        print(f"\n{'='*60}")
        print("Fix Summary")
        print(f"{'='*60}")
        print(f"Total samples:       {total}")
        print(f"OK (original):       {original_ok}")
        print(f"OK (after fix):      {fixed_ok}")
        print(f"Fix failed:          {failed}")
        print(f"Skipped (non-dim):   {skipped_dim}")
        print(f"Skipped (resources): {skipped_res}")
        if total:
            rate = (original_ok + fixed_ok) / total * 100
            print(f"Success rate:        {rate:.1f}%")
        print(f"{'='*60}")
        print("Preprocessing Stats:")
        print(f"  Auto-skipped:      {preprocessed_skip}")
        print(f"  Auto-fix success:  {preprocessed_fix_ok}")
        print(f"  Auto-fix failed:   {preprocessed_fix_failed}")
        print(f"  API fix success:   {api_fix_ok}")
        print(f"{'='*60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically fix dimension errors in dimension-generalized samples using the Anthropic API (Claude)"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=None,
        help="Root directory of samples to fix, each subdirectory is a sample",
    )
    parser.add_argument(
        "--sample-list-with-status",
        type=str,
        default=None,
        help="Path to sample list file with status, format: STATUS<TAB>sample_absolute_path per line",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to process, 0 means all (default: 0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum fix retry attempts per sample (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each run_model execution (default: 600)",
    )
    parser.add_argument(
        "--original-samples-dir",
        type=str,
        default="samples/",
        help="Root directory of original (non-dimension-generalized) samples (default: samples/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="Claude model name (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for logs and reports (default: current directory)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Anthropic API base URL (required)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Anthropic API Key (required)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for parallel run_model execution, e.g. '0,1,2,3' (default: sequential)",
    )
    parser.add_argument(
        "--custom-header",
        type=str,
        required=True,
        help="Custom header value for API requests (required)",
    )

    args = parser.parse_args()

    if not args.samples_dir and not args.sample_list_with_status:
        parser.error("Must provide --samples-dir or --sample-list-with-status")

    return args


def main():
    args = parse_args()
    orchestrator = MainOrchestrator(args)
    orchestrator.run()


if __name__ == "__main__":
    main()
