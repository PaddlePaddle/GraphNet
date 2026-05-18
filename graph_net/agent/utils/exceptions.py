"""Custom exception classes for Agent.

Each exception may carry an `error_category` so that
error_classifier.py can route without string matching.
"""

from enum import Enum
from typing import Optional


class GraphExtractionErrorCategory(str, Enum):
    """Known categories of extraction failure."""

    # Pre-extraction failures
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_FORBIDDEN = "model_forbidden"
    MODEL_DOWNLOAD_ERROR = "model_download_error"

    # Config / metadata analysis failures
    CONFIG_NOT_FOUND = "config_not_found"
    CONFIG_PARSE_ERROR = "config_parse_error"
    METADATA_ANALYSIS_FAILED = "metadata_analysis_failed"

    # Script generation failures
    CODE_GEN_ERROR = "code_gen_error"

    # Script execution failures
    SCRIPT_EXECUTION_FAILED = "script_execution_failed"
    SCRIPT_TIMEOUT = "script_timeout"
    OUTPUT_DIR_NOT_FOUND = "output_dir_not_found"

    # LLM retry failures
    LLM_TIMEOUT = "llm_timeout"
    LLM_EXIT_ERROR = "llm_exit_error"

    # Post-extraction failures
    SAMPLE_INCOMPLETE = "sample_incomplete"
    FORWARD_VERIFY_FAILED = "forward_verify_failed"
    VERIFICATION_TIMEOUT = "verification_timeout"
    VERIFICATION_FAILED = "verification_failed"

    # Catch-all
    UNKNOWN = "unknown"


class AgentError(Exception):
    """Base exception for Agent errors.

    Subclasses can set `default_category` so that raise-sites do not
    need to repeat the category when the default is sufficient.
    """

    default_category: Optional[GraphExtractionErrorCategory] = None

    def __init__(
        self,
        message: str,
        error_category: Optional[GraphExtractionErrorCategory] = None,
    ):
        super().__init__(message)
        self.error_category = error_category or self.default_category


class ModelFetchError(AgentError):
    """Raised when model fetching fails.

    Default: MODEL_DOWNLOAD_ERROR.
    Raise-sites should override for 404 (MODEL_NOT_FOUND)
    or 403 (MODEL_FORBIDDEN).
    """

    default_category = GraphExtractionErrorCategory.MODEL_DOWNLOAD_ERROR


class MetadataAnalysisError(AgentError):
    """Raised when model metadata/config analysis fails.

    Covers config missing, JSON parse errors, and unsupported architectures.
    """

    default_category = GraphExtractionErrorCategory.METADATA_ANALYSIS_FAILED


class CodeGenerationError(AgentError):
    """Raised when code generation fails.

    Default: CODE_GEN_ERROR.
    Raise-sites should override for LLM-specific failures
    (LLM_TIMEOUT / LLM_EXIT_ERROR).
    """

    default_category = GraphExtractionErrorCategory.CODE_GEN_ERROR


class GraphExtractionError(AgentError):
    """Raised when graph extraction fails.

    Default: UNKNOWN — raise-sites MUST override with one of:
    - SCRIPT_EXECUTION_FAILED
    - SCRIPT_TIMEOUT
    - OUTPUT_DIR_NOT_FOUND
    """

    default_category = GraphExtractionErrorCategory.UNKNOWN


class SampleVerificationError(AgentError):
    """Raised when sample verification fails.

    Default: VERIFICATION_FAILED.
    Raise-sites may override with VERIFICATION_TIMEOUT.
    """

    default_category = GraphExtractionErrorCategory.VERIFICATION_FAILED
