from graph_net.positive_tolerance_interpretation import PositiveToleranceInterpretation


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
        mapping = {
            1: "accuracy",
            2: "runtime_fail",
            3: "compile_fail"
        }
        return mapping.get(errno, "unknown_error")

    def get_tolerance_mapping(self) -> dict[int, int]:
        return {
            1: 1,  # Accuracy -> t >= 1
            2: 3,  # Runtime  -> t >= 3
            3: 3  # Compile  -> t >= 3
        }

    def is_error_tolerated(self, tolerance: int, base_error_code: str) -> bool:
        if base_error_code == "correct":
            return True

        # Eager/Reference failures are never tolerated
        if base_error_code in ["eager_fail", "reference_fail"]:
            return False

        if tolerance >= 3:
            return True
        elif base_error_code == "accuracy" and tolerance >= 1:
            return True

        return False