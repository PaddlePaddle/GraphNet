from typing import Type, List
from graph_net.positive_tolerance_interpretation import PositiveToleranceInterpretation
from graph_net.default_positive_tolerance_interpretation import (
    DefaultPositiveToleranceInterpretation,
)
from graph_net.mismatch_extended_positive_tolerance_interpretation import (
    MismatchExtendedPositiveToleranceInterpretation,
)


def get_positive_tolerance_interpretation(
    type_name: str,
) -> PositiveToleranceInterpretation:
    """
    Factory function to retrieve an instance of the requested interpretation strategy.

    Args:
        type_name: The string identifier (e.g., 'default', 'mismatch_extended')

    Returns:
        An instance of PositiveToleranceInterpretation.

    Raises:
        ValueError: If type_name is not registered.
    """
    if type_name not in _g_type_name2_positive_tolerance_interpretation_cls:
        supported = list(_g_type_name2_positive_tolerance_interpretation_cls.keys())
        raise ValueError(
            f"Unknown positive tolerance interpretation: '{type_name}'. Supported: {supported}"
        )

    # Instantiate and return.
    # If stateful caching is needed, this logic can be modified to return singletons.
    cls = _g_type_name2_positive_tolerance_interpretation_cls[type_name]
    return cls()


def get_supported_positive_tolerance_interpretation_types() -> List[str]:
    return list(_g_type_name2_positive_tolerance_interpretation_cls.keys())


# Registry of available classes
_g_type_name2_positive_tolerance_interpretation_cls: dict[
    str, Type[PositiveToleranceInterpretation]
] = {
    "default": DefaultPositiveToleranceInterpretation,
    "mismatch_extended": MismatchExtendedPositiveToleranceInterpretation,
}
