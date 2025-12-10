from graph_net.positive_tolerance_interpretation import PositiveToleranceInterpretation


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
        if not error_type:
            return 3
        etype = error_type.lower()

        if "accuracy" in etype:
            return 1
        if any(k in etype for k in ["nan", "inf", "type_mismatch", "shape_mismatch"]):
            return 2
        if "compile_fail" in etype:
            return 4
        return 3

    def get_error_type(self, errno: int) -> str:
        mapping = {
            1: "accuracy",
            2: "type/shape_mismatch",
            3: "runtime_fail",
            4: "compile_fail"
        }
        return mapping.get(errno, "unknown_error")

    def get_tolerance_mapping(self) -> dict[int, int]:
        return {
            1: 1,
            2: 2,
            3: 3,
            4: 4
        }

    def is_error_tolerated(self, tolerance: int, base_error_code: str) -> bool:
        if base_error_code == "correct":
            return True

        if base_error_code in ["eager_fail", "reference_fail"]:
            return False

        # Level 4: Compile
        if tolerance >= 4 and base_error_code == "compile_fail":
            return True

        # Level 3: Runtime
        if tolerance >= 3 and base_error_code in ["runtime_fail", "execution_fail"]:
            return True

        # Level 2: Data/Type
        if tolerance >= 2 and base_error_code in ["nan", "inf", "type_mismatch", "shape_mismatch"]:
            return True

        # Level 1: Accuracy
        if tolerance >= 1 and base_error_code == "accuracy":
            return True

        return False