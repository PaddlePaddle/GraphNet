import os
from datetime import datetime


class FixLogger:
    """Write per-sample fix_log.md and summary report."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def write_sample_log(
        self,
        sample_name: str,
        sample_path: str,
        status: str,
        retry_count: int,
        attempts: list,
        final_output_summary: str,
    ):
        """Write per-sample fix_log.md."""
        sample_log_dir = os.path.join(self.log_dir, sample_name)
        os.makedirs(sample_log_dir, exist_ok=True)
        log_path = os.path.join(sample_log_dir, "fix_log.md")

        lines = [
            f"# {sample_name} Fix Log",
            "",
            f"- **Sample Path**: {sample_path}",
            f"- **Status**: {status}",
            f"- **Fix Count**: {retry_count}",
            "",
        ]

        if not attempts:
            lines.append("## Change Log")
            lines.append("")
            lines.append("No fix needed")
            lines.append("")
        else:
            lines.append("## Change Log")
            lines.append("")
            for i, attempt in enumerate(attempts, 1):
                lines.append(f"### Attempt {i}")
                lines.append("")
                lines.append(
                    f"- **Error Message**: {attempt.get('error_summary', 'N/A')}"
                )
                if attempt.get("action") == "skip_sample":
                    lines.append(f"- **Skip Reason**: {attempt.get('reason', 'N/A')}")
                else:
                    for edit in attempt.get("edits", []):
                        lines.append(f"- **Fixed File**: {edit.get('file', 'N/A')}")
                        lines.append("- **Original Code**:")
                        lines.append("```python")
                        lines.append(edit.get("old", ""))
                        lines.append("```")
                        lines.append("- **Changed To**:")
                        lines.append("```python")
                        lines.append(edit.get("new", ""))
                        lines.append("```")
                    if attempt.get("summary"):
                        lines.append(f"- **Fix Reason**: {attempt['summary']}")
                lines.append("")

        lines.append("## Final Status")
        lines.append("")
        if status.startswith("OK"):
            lines.append("- **Exit Code**: 0")
        else:
            lines.append("- **Exit Code**: Non-zero")
        lines.append(f"- **Final Output**: {final_output_summary}")
        lines.append("")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def write_report(
        self,
        samples_dir: str,
        results: list,
    ):
        """Write summary report fix_samples_report.md."""
        report_path = os.path.join(self.log_dir, "fix_samples_report.md")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total = len(results)
        original_ok = sum(1 for r in results if r["status"] == "OK(original)")
        fixed_ok = sum(1 for r in results if r["status"] == "OK(fixed)")
        failed = sum(1 for r in results if r["status"] == "Failed")
        skipped_dim = sum(1 for r in results if r["status"] == "Skipped(non-dim error)")
        skipped_res = sum(
            1 for r in results if r["status"] == "Skipped(insufficient resources)"
        )
        success_rate = (
            f"{(original_ok + fixed_ok) / total * 100:.1f}%" if total else "N/A"
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

        lines = [
            "# Sample Auto-Fix Report",
            "",
            f"- **Sample Directory**: {samples_dir}",
            f"- **Execution Time**: {now}",
            f"- **Max Retries**: {max(r.get('max_retries', 0) for r in results) if results else 0}",
            "",
            "## Summary Statistics",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total Samples | {total} |",
            f"| OK (original) | {original_ok} |",
            f"| OK (after fix) | {fixed_ok} |",
            f"| Fix Failed | {failed} |",
            f"| Skipped (non-dim error) | {skipped_dim} |",
            f"| Skipped (insufficient resources) | {skipped_res} |",
            "",
            f"Success Rate: {success_rate}",
            "",
            "## Preprocessing Statistics",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Preprocessed auto-skip | {preprocessed_skip} |",
            f"| Preprocessed auto-fix success | {preprocessed_fix_ok} |",
            f"| Preprocessed auto-fix then failed | {preprocessed_fix_failed} |",
            f"| API fix success | {api_fix_ok} |",
            "",
            "## Sample Results",
            "",
            "| Sample Name | Status | Fix Count | Log Path |",
            "|-------------|--------|-----------|----------|",
        ]

        for r in results:
            log_rel = os.path.relpath(
                os.path.join(self.log_dir, r["sample_name"], "fix_log.md")
            )
            lines.append(
                f"| {r['sample_name']} | {r['status']} | {r['retry_count']} | {log_rel} |"
            )

        # Failed samples list
        failed_samples = [r for r in results if r["status"] == "Failed"]
        if failed_samples:
            lines.append("")
            lines.append("## Failed Samples")
            lines.append("")
            lines.append("| Sample Name | Last Error Type | Last Error Summary |")
            lines.append("|-------------|-----------------|--------------------|")
            for r in failed_samples:
                err_summary = r.get("last_error_summary", "N/A")
                err_type = r.get("last_error_type", "Unknown")
                # Truncate long error messages
                if len(err_summary) > 80:
                    err_summary = err_summary[:77] + "..."
                lines.append(f"| {r['sample_name']} | {err_type} | {err_summary} |")

        # Skipped samples list (non-dimension error)
        skipped_dim_samples = [
            r for r in results if r["status"] == "Skipped(non-dim error)"
        ]
        if skipped_dim_samples:
            lines.append("")
            lines.append("## Skipped Samples (Non-Dimension Error)")
            lines.append("")
            lines.append("| Sample Name | Error Type | Error Summary |")
            lines.append("|-------------|------------|---------------|")
            for r in skipped_dim_samples:
                err_summary = r.get("skip_reason", "N/A")
                if len(err_summary) > 80:
                    err_summary = err_summary[:77] + "..."
                lines.append(f"| {r['sample_name']} | Non-dim error | {err_summary} |")

        # Skipped samples list (insufficient resources)
        skipped_res_samples = [
            r for r in results if r["status"] == "Skipped(insufficient resources)"
        ]
        if skipped_res_samples:
            lines.append("")
            lines.append("## Skipped Samples (Insufficient Resources)")
            lines.append("")
            lines.append("| Sample Name | Skip Reason |")
            lines.append("|-------------|-------------|")
            for r in skipped_res_samples:
                lines.append(f"| {r['sample_name']} | {r.get('skip_reason', 'N/A')} |")

        lines.append("")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
