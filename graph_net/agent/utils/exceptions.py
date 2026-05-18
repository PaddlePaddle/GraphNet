"""Custom exception classes for Agent.

Each exception may carry an `error_category` string so that
error_classifier.py can route without string matching.
"""

from typing import Optional


class AgentError(Exception):
    """Base exception for Agent errors.

    Subclasses can set `default_category` so that raise-sites do not
    need to repeat the category when the default is sufficient.
    """

    default_category: Optional[str] = None

    def __init__(
        self,
        message: str,
        error_category: Optional[str] = None,
    ):
        super().__init__(message)
        self.error_category = error_category or self.default_category


class ModelFetchError(AgentError):
    """Raised when model fetching fails.

    Default: model_download_error.
    Raise-sites should override for 404 (model_not_found)
    or 403 (model_forbidden).
    """

    default_category = "model_download_error"


class MetadataAnalysisError(AgentError):
    """Raised when model metadata/config analysis fails.

    Covers config missing, JSON parse errors, and unsupported architectures.
    """

    default_category = "metadata_analysis_error"


class CodeGenerationError(AgentError):
    """Raised when code generation fails.

    Default: code_gen_error.
    Raise-sites should override for LLM-specific failures
    (llm_timeout / llm_exit_error).
    """

    default_category = "code_gen_error"


class GraphExtractionError(AgentError):
    """Raised when graph extraction fails.

    Default: unknown — raise-sites MUST override with one of:
    - script_execution_failed
    - script_timeout
    - output_dir_not_found
    """

    default_category = "unknown"


class SampleVerificationError(AgentError):
    """Raised when sample verification fails.

    Default: verification_failed.
    Raise-sites may override with verification_timeout.
    """

    default_category = "verification_failed"
