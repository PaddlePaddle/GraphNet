"""Custom exception classes for Agent"""


class AgentError(Exception):
    """Base exception for Agent errors"""

    pass


class ModelFetchError(AgentError):
    """Raised when model fetching fails"""

    pass


class AnalysisError(AgentError):
    """Raised when model analysis fails"""

    pass


class CodeGenError(AgentError):
    """Raised when code generation fails"""

    pass


class ExtractionError(AgentError):
    """Raised when graph extraction fails"""

    pass


class VerificationError(AgentError):
    """Raised when sample verification fails"""

    pass
