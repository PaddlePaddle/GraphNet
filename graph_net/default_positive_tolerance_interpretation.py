from enum import IntEnum

from graph_net.positive_tolerance_interpretation import PositiveToleranceInterpretation


class DefaultErrorEnum(IntEnum):
    """
    Values correspond to the minimum tolerance level required.
    """

    kAccuracyViolation = 1
    kValueTypeOrMetaMismatch = 3
    kExecutionFailed = 3
    kCompilationFailed = 3

    @classmethod
    def get_error_enum(cls, base_error_type: str) -> "DefaultErrorEnum":
        if not base_error_type:
            return cls.kExecutionFailed

        etype = base_error_type.lower()

        if "accuracy" in etype:
            return cls.kAccuracyViolation

        if any(x in etype for x in ["nan", "inf", "type_mismatch", "shape_mismatch"]):
            return cls.kValueTypeOrMetaMismatch

        if "compile_fail" in etype:
            return cls.kCompilationFailed

        return cls.kExecutionFailed


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
        if not error_type:
            return 2
        etype = error_type.lower()
        if "accuracy" in etype:
            return 1
        if "compile_fail" in etype:
            return 3
        return 2

    def get_error_type(self, errno: int) -> str:
        mapping = {1: "accuracy", 2: "runtime_fail", 3: "compile_fail"}
        return mapping.get(errno, "unknown_error")

    def get_tolerance_mapping(self) -> dict[int, int]:
        return {
            1: 1,  # Accuracy -> t >= 1
            2: 3,  # Runtime  -> t >= 3
            3: 3,  # Compile  -> t >= 3
        }

    def is_error_tolerated(self, tolerance: int, base_error_code: str) -> bool:
        if base_error_code == "correct":
            return True
        if base_error_code in ["eager_fail", "reference_fail"]:
            return False

        try:
            error_level = DefaultErrorEnum.get_error_enum(base_error_code)
            return tolerance >= error_level.value
        except (ValueError, KeyError):
            return False
