from enum import IntEnum

from graph_net.positive_tolerance_interpretation import PositiveToleranceInterpretation


class MismatchExtendedErrorEnum(IntEnum):
    """
    Values correspond to the minimum tolerance level required.
    """

    kAccuracyViolation = 1
    kValueTypeOrMetaMismatch = 2
    kExecutionFailed = 3
    kCompilationFailed = 4

    @classmethod
    def get_error_enum(cls, base_error_type: str) -> "MismatchExtendedErrorEnum":
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


class MismatchExtendedPositiveToleranceInterpretation(PositiveToleranceInterpretation):
    """
    Extended interpretation (ESt):
    - t=1: Accuracy
    - t=2: NaN/Inf/Type/Shape
    - t=3: Runtime
    - t=4: Compile
    """

    def __init__(self, *argc, **kwargs):
        super().__init__(*argc, **kwargs)

    def type_name(self) -> str:
        return "mismatch_extended"

    def get_errno(self, error_type: str) -> int:
        return MismatchExtendedErrorEnum.get_error_enum(error_type).value

    def get_error_type(self, errno: int) -> str:
        mapping = {
            1: "accuracy",
            2: "type/shape_mismatch",
            3: "runtime_fail",
            4: "compile_fail",
        }
        return mapping.get(errno, "unknown_error")

    def get_tolerance_mapping(self) -> dict[int, int]:
        return {
            MismatchExtendedErrorEnum.kAccuracyViolation.value: 1,
            MismatchExtendedErrorEnum.kValueTypeOrMetaMismatch.value: 2,
            MismatchExtendedErrorEnum.kExecutionFailed.value: 3,
            MismatchExtendedErrorEnum.kCompilationFailed.value: 4,
        }

    def is_error_tolerated(self, tolerance: int, base_error_code: str) -> bool:
        if base_error_code == "correct":
            return True
        if base_error_code in ["eager_fail", "reference_fail"]:
            return False

        error_enum = MismatchExtendedErrorEnum.get_error_enum(base_error_code)
        mapping = self.get_tolerance_mapping()
        required_threshold = mapping.get(error_enum.value, 999)

        return tolerance >= required_threshold

    def num_errno_enum_values(self) -> int:
        """
        Extended mode defines 4 levels of errors:
        1: Accuracy
        2: Type/Shape/NaN
        3: Runtime
        4: Compilation
        """
        return len(MismatchExtendedErrorEnum)
