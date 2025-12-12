from enum import IntEnum

from graph_net.positive_tolerance_interpretation import PositiveToleranceInterpretation


class DefaultErrorEnum(IntEnum):
    """
    Values correspond to the minimum tolerance level required.
    """

    kAccuracyViolation = 1  # Accuracy
    kRuntimeFailure = 2  # Includes Runtime, NaN, Inf, TypeMismatch, etc.
    kCompilationFailed = 3  # Compile Failure

    @classmethod
    def get_error_enum(cls, base_error_type: str) -> "DefaultErrorEnum":
        if not base_error_type:
            return cls.kRuntimeFailure

        etype = base_error_type.lower()

        if "accuracy" in etype:
            return cls.kAccuracyViolation

        if "compile_fail" in etype:
            return cls.kCompilationFailed

        return cls.kRuntimeFailure


class DefaultPositiveToleranceInterpretation(PositiveToleranceInterpretation):
    """
    Legacy interpretation:
    - t=1: Accuracy errors tolerated.
    - t=3: Runtime/Compilation errors tolerated.
    """

    def __init__(self, *argc, **kwargs):
        super().__init__(*argc, **kwargs)

    def type_name(self) -> str:
        return "default"

    def get_errno(self, error_type: str) -> int:
        return DefaultErrorEnum.get_error_enum(error_type).value

    def get_error_type(self, errno: int) -> str:
        mapping = {1: "accuracy", 2: "runtime_fail", 3: "compile_fail"}
        return mapping.get(errno, "unknown_error")

    def get_tolerance_mapping(self) -> dict[int, int]:
        return {
            DefaultErrorEnum.kAccuracyViolation.value: 1,
            DefaultErrorEnum.kRuntimeFailure.value: 3,
            DefaultErrorEnum.kCompilationFailed.value: 3,
        }

    def is_error_tolerated(self, tolerance: int, base_error_code: str) -> bool:
        if base_error_code == "correct":
            return True
        if base_error_code in ["eager_fail", "reference_fail"]:
            return False

        error_enum = DefaultErrorEnum.get_error_enum(base_error_code)
        mapping = self.get_tolerance_mapping()
        required_threshold = mapping.get(error_enum.value, 999)

        return tolerance >= required_threshold

    def num_errno_enum_values(self) -> int:
        """
        Default mode defines 3 levels of errors:
        1: Accuracy
        2: Runtime (Generic)
        3: Compilation
        """
        return len(DefaultErrorEnum)
