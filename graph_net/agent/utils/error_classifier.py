"""Error classification for extraction failures.

Classification is driven entirely by the exception's `error_category`
attribute (set at the raise-site).  No string keyword matching is
performed here — keywords belong in the code that raises the exception.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from graph_net.agent.utils.exceptions import GraphExtractionErrorCategory


@dataclass
class ErrorRecord:
    """Single error occurrence."""

    model_id: str
    category: GraphExtractionErrorCategory
    message: str


class GraphExtractionErrorClassifier:
    """Classify extraction errors and keep per-model records.

    Usage:
        classifier = GraphExtractionErrorClassifier()
        category = classifier.classify_from_exception(exc)
        classifier.record(model_id, category, str(exc))

        # After run
        report = classifier.summary()
    """

    def __init__(self):
        self.records: List[ErrorRecord] = []
        self._by_model: Dict[str, ErrorRecord] = {}

    @staticmethod
    def classify_from_exception(exc: Exception) -> GraphExtractionErrorCategory:
        """Classify from the exception's ``error_category`` attribute.

        Falls back to UNKNOWN when the attribute is missing or invalid.
        """
        raw = getattr(exc, "error_category", None)
        if raw is not None:
            try:
                return GraphExtractionErrorCategory(raw)
            except ValueError:
                pass
        return GraphExtractionErrorCategory.UNKNOWN

    def record(
        self,
        model_id: str,
        category: GraphExtractionErrorCategory,
        message: str,
    ) -> None:
        """Store one error record."""
        rec = ErrorRecord(model_id=model_id, category=category, message=message)
        self.records.append(rec)
        self._by_model[model_id] = rec

    def classify_and_record(
        self,
        model_id: str,
        exception: Exception,
    ) -> GraphExtractionErrorCategory:
        """Convenience: classify from exception and store."""
        category = self.classify_from_exception(exception)
        self.record(model_id, category, str(exception))
        return category

    def get_record(self, model_id: str) -> Optional[ErrorRecord]:
        return self._by_model.get(model_id)

    def get_models_by_category(
        self, category: GraphExtractionErrorCategory
    ) -> List[str]:
        return [rec.model_id for rec in self.records if rec.category == category]

    def summary(self) -> Dict[str, object]:
        counts: Dict[str, int] = defaultdict(int)
        per_category: Dict[str, List[str]] = defaultdict(list)
        for rec in self.records:
            cat_name = rec.category.value
            counts[cat_name] += 1
            per_category[cat_name].append(rec.model_id)
        return {
            "total_errors": len(self.records),
            "category_counts": dict(counts),
            "models_per_category": dict(per_category),
        }

    def markdown_report(self) -> str:
        lines = ["# Extraction Error Report", ""]
        lines.append(f"**Total errors**: {len(self.records)}")
        lines.append("")

        counts: Dict[GraphExtractionErrorCategory, int] = defaultdict(int)
        per_cat: Dict[GraphExtractionErrorCategory, List[ErrorRecord]] = defaultdict(
            list
        )
        for rec in self.records:
            counts[rec.category] += 1
            per_cat[rec.category].append(rec)

        lines.append("## Summary by Category")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat.value} | {cnt} |")
        lines.append("")

        lines.append("## Details")
        lines.append("")
        for cat, recs in sorted(per_cat.items(), key=lambda x: -len(x[1])):
            lines.append(f"### {cat.value} ({len(recs)})")
            lines.append("")
            for rec in recs[:10]:
                msg = (
                    rec.message[:120] + "..." if len(rec.message) > 120 else rec.message
                )
                lines.append(f"- `{rec.model_id}`: {msg}")
            if len(recs) > 10:
                lines.append(f"- ... and {len(recs) - 10} more")
            lines.append("")

        return "\n".join(lines)
