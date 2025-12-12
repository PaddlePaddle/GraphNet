from abc import ABC, abstractmethod


class PositiveToleranceInterpretation(ABC):
    """
    Abstract base class defining how positive tolerance values (t > 0)
    are interpreted and mapped to specific error types.
    """

    def __init__(self, *argc, **kwargs):
        pass

    @abstractmethod
    def type_name(self) -> str:
        """Return the unique string identifier for this interpretation strategy."""
        raise NotImplementedError

    @abstractmethod
    def get_errno(self, error_type: str) -> int:
        """Map a raw error type string to an internal error number (errno)."""
        raise NotImplementedError

    @abstractmethod
    def get_error_type(self, errno: int) -> str:
        """Map an internal error number (errno) back to a representative string."""
        raise NotImplementedError

    @abstractmethod
    def get_tolerance_mapping(self) -> dict[int, int]:
        """
        Return the mapping of errno.
        Used for statistical calculations (Gamma/Pi).
        """
        raise NotImplementedError

    @abstractmethod
    def is_error_tolerated(self, tolerance: int, base_error_code: str) -> bool:
        """
        Determine if a specific error is considered 'correct' under the given tolerance.
        Replaces the old 'fake_perf_degrad' logic.
        """
        raise NotImplementedError

    @abstractmethod
    def num_errno_enum_values(self) -> int:
        """
        Return the number of defined error categories (or the maximum errno).

        Example:
            - Default: returns 3 (Accuracy, Runtime, Compile)
            - MismatchExtended: returns 4 (Accuracy, Data, Runtime, Compile)
        """
        raise NotImplementedError
