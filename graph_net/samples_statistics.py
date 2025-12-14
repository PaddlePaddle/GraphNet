"""
Aggregated statistics calculation module for S(t) and ES(t) metrics.

This module provides independent functions for calculating each aggregated parameter
(alpha, beta, gamma, lambda, eta, pi) according to Appendix B and C of the paper.
"""
from typing import Union, Optional

from scipy.stats import gmean
from collections.abc import Callable
from graph_net.positive_tolerance_interpretation import PositiveToleranceInterpretation


def get_errno_from_error_type(
    error_type: str, positive_tolerance_interpretation: PositiveToleranceInterpretation
) -> int:
    """
    Map error type string to errno (error number) using the appropriate strategy.

    Args:
        error_type: Error type string (e.g., "accuracy", "runtime_fail")
        positive_tolerance_interpretation: Evaluation mode ("default" or "mismatch_extended")

    Returns:
        int: Errno based on the selected positive_tolerance_interpretation's logic.
    """
    return positive_tolerance_interpretation.get_errno(error_type)


def get_errno_tolerance_mapping(
    custom_mapping, positive_tolerance_interpretation: PositiveToleranceInterpretation
):
    """
    Map errno (error number) back to error type string.

    This is the reverse mapping of get_errno_from_error_type.
    Used when error type string information is needed.

    Args:
        errno: Error number
        positive_tolerance_interpretation: Evaluation mode ("default" or "mismatch_extended")

    Returns:
        Representative error type string (e.g., "accuracy", "compile_fail")
    """
    if custom_mapping:
        return custom_mapping
    return positive_tolerance_interpretation.get_tolerance_mapping()


def calculate_alpha(correct_speedups: list[float]) -> float:
    """
    Calculate alpha: geometric mean of correct sample speedups.

    According to Appendix B: alpha is the geometric mean of all correct sample speedups.

    Args:
        correct_speedups: List of speedup values for correct samples

    Returns:
        Alpha value (geometric mean), or 1.0 if list is empty
    """
    return gmean(correct_speedups) if len(correct_speedups) > 0 else 1.0


def calculate_beta(correct_speedups: list[float]) -> float:
    """
    Calculate beta: geometric mean of slowdown sample speedups.

    According to Appendix B: beta is the geometric mean of speedups for samples
    that are correct but have speedup < 1 (slowdown cases).

    Args:
        correct_speedups: List of speedup values for correct samples

    Returns:
        Beta value (geometric mean of slowdown cases), or 1.0 if no slowdown cases
    """
    slowdown_speedups = list(filter(lambda x: x < 1, correct_speedups))
    return gmean(slowdown_speedups) if len(slowdown_speedups) > 0 else 1.0


def calculate_lambda(correct_speedups: list[float], total_samples: int) -> float:
    """
    Calculate lambda: fraction of correct samples.

    According to Appendix B: lambda = M / N, where M is correct count and N is total samples.

    Args:
        correct_speedups: List of speedup values for correct samples
        total_samples: Total number of samples

    Returns:
        Lambda value (fraction of correct samples), or 1.0 if total_samples is 0 (lenient handling)
    """
    correct_count = len(correct_speedups)
    return correct_count / total_samples if total_samples > 0 else 1.0


def calculate_eta(correct_speedups: list[float]) -> float:
    """
    Calculate eta: fraction of slowdown cases within correct samples.

    According to Appendix B: eta = K / M, where K is the number of slowdown cases
    within correct samples, and M is the total correct count.

    Args:
        correct_speedups: List of speedup values for correct samples

    Returns:
        Eta value (fraction of slowdown cases), or 0.0 if no correct samples
    """
    correct_count = len(correct_speedups)
    if correct_count == 0:
        return 0.0
    slowdown_speedups = list(filter(lambda x: x < 1, correct_speedups))
    correct_negative_speedup_count = len(slowdown_speedups)
    return correct_negative_speedup_count / correct_count


def calculate_pi(
    errno2count: dict[Union[int, str], int],
    total_samples: int,
    correct_speedups: list[float],
) -> dict[Union[int, str], float]:
    """
    Calculate pi: error type proportions for t > 0.

    According to Appendix C: pi_c is the proportion of error type c among all error samples.

    Args:
        errno2count: Dictionary mapping errno (error number) to their counts.
            Errno values: 1=accuracy, 2=runtime, 3=compilation.
        total_samples: Total number of samples
        correct_speedups: List of speedup values for correct samples

    Returns:
        Dictionary mapping errno to their proportions among error samples.
        If error_count is 0, returns a dictionary with all proportions set to 0.0.
    """
    correct_count = len(correct_speedups)
    error_count = total_samples - correct_count
    counted_errors = sum(errno2count.values())
    assert (
        error_count == counted_errors
    ), f"error_count mismatch: got {error_count}, but errno2count sums to {counted_errors}"
    if error_count == 0:
        return {errno: 0.0 for errno in errno2count.keys()}

    pi = {}
    for errno, count in errno2count.items():
        pi[errno] = count / error_count
    return pi


def resolve_errno_tolerance(
    errno2count: dict[Union[int, str], int],
    positive_tolerance_interpretation: PositiveToleranceInterpretation,
    errno_tolerance_overrides: Optional[dict[Union[int, str], int]] = None,
) -> dict[Union[int, str], int]:
    """
    Build a sorted errno -> tolerance map for downstream gamma calculation.

    Args:
        errno2count: Observed errno occurrences in the dataset. Keys can be int (default) or str (mismatch_extended).
        errno_tolerance_overrides: Optional overrides mapping errno to its minimal tolerated tolerance.
                                   In extended mode, this should contain the full mapping (e.g., "nan": 2).

    Returns:
        Ordered dict (by errno) mapping each errno seen in errno2count
        to the tolerance value where it becomes tolerated.

        Defaults logic (if not in overrides):
        - int 1 (accuracy) -> 1
        - int others -> 3
        - str (unknown) -> 999 (Treat as severe error if not explicitly mapped)
    """
    errno_tolerance_overrides = errno_tolerance_overrides or {}

    base_mapping = positive_tolerance_interpretation.get_tolerance_mapping()

    def tolerance_for(err_key: Union[int, str]) -> int:
        if err_key in errno_tolerance_overrides:
            return errno_tolerance_overrides[err_key]

        errno_id = None
        if isinstance(err_key, int):
            errno_id = err_key
        elif isinstance(err_key, str):
            errno_id = positive_tolerance_interpretation.get_errno(err_key)

        if errno_id is not None and errno_id in base_mapping:
            return base_mapping[errno_id]

        return 999

    sorted_keys = sorted(errno2count.keys(), key=lambda x: str(x))

    return {errno: tolerance_for(errno) for errno in sorted_keys}


def calculate_gamma(
    tolerance: int,
    pi_value4errno: Callable[[int], float],
    errno2tolerance: dict[int, int],
    b: float = 0.1,
) -> float:
    """
    Calculate gamma_t: average error penalty factor.

    According to Appendix C: gamma_t = b^(sum(π_c * indicator(t < threshold_c)))
    where indicator(t < threshold_c) = 1 if error type c is not tolerated at tolerance t, else 0.

    Args:
        tolerance: Tolerance level t
        pi_value4errno: Function that takes errno and returns π_c (proportion of error type c).
        errno2tolerance: Mapping of errno to tolerance thresholds.
            An error type is tolerated (not penalized) when t >= threshold for that errno.
        b: Base penalty for severe errors (default: 0.1)

    Returns:
        Gamma value (average error penalty)
    """
    if tolerance <= 0:
        return b

    # Calculate indicator-weighted pi sum for errnos that are not tolerated
    pi_sum = sum(
        pi_value
        for errno, errno_tolerance in errno2tolerance.items()
        for pi_value in [pi_value4errno(errno)]
        if tolerance < errno_tolerance
    )

    return b**pi_sum


def calculate_s_t_from_aggregated(
    alpha: float,
    beta: float,
    lambda_: float,
    eta: float,
    negative_speedup_penalty: float,
    b: float,
) -> float:
    """
    Calculate S(t) from aggregated parameters.

    According to Appendix B: S_t = α^λ · β^(ληp) · b^(1-λ)

    Args:
        alpha: Geometric mean speedup of correct samples
        beta: Geometric mean speedup of slowdown cases
        lambda_: Fraction of correct samples
        eta: Fraction of slowdown cases within correct samples
        negative_speedup_penalty: Penalty power p for negative speedup
        b: Base penalty for severe errors or accuracy violation

    Returns:
        S(t) value calculated from aggregated parameters
    """
    return (
        alpha**lambda_
        * beta ** (lambda_ * eta * negative_speedup_penalty)
        * b ** (1 - lambda_)
    )


def calculate_es_t_from_aggregated(
    alpha: float,
    beta: float,
    lambda_: float,
    eta: float,
    gamma: float,
    negative_speedup_penalty: float,
) -> float:
    """
    Calculate ES(t) from aggregated parameters.

    According to Appendix C: ES_t = α^λ · β^(ληp) · γ_t^(1-λ)

    Args:
        alpha: Geometric mean speedup of correct samples
        beta: Geometric mean speedup of slowdown cases
        lambda_: Fraction of correct samples
        eta: Fraction of slowdown cases within correct samples
        gamma: Average error penalty factor
        negative_speedup_penalty: Penalty power p for negative speedup

    Returns:
        ES(t) value calculated from aggregated parameters
    """
    return (
        alpha**lambda_
        * beta ** (lambda_ * eta * negative_speedup_penalty)
        * gamma ** (1 - lambda_)
    )


def calculate_es_components_values(
    total_samples: int,
    correct_speedups: list[float],
    errno2count: dict[Union[int, str], int],  # support str
    tolerance: int,
    positive_tolerance_interpretation: PositiveToleranceInterpretation,
    negative_speedup_penalty: float = 0.0,
    b: float = 0.1,
    pi: Optional[dict[Union[int, str], float]] = None,
    errno_to_tolerance: Optional[dict[Union[int, str], int]] = None,
) -> dict:
    """
    Calculate aggregated parameters for a given tolerance level.
    """

    if pi is None:
        pi = calculate_pi(errno2count, total_samples, correct_speedups)

    errno_to_tolerance = get_errno_tolerance_mapping(
        errno_to_tolerance, positive_tolerance_interpretation
    )

    errno2tolerance = resolve_errno_tolerance(
        errno2count, positive_tolerance_interpretation, errno_to_tolerance
    )

    def pi_value4errno(errno: Union[int, str]) -> float:
        return pi.get(errno, 0.0)

    alpha = calculate_alpha(correct_speedups)
    beta = calculate_beta(correct_speedups)
    lambda_ = calculate_lambda(correct_speedups, total_samples)
    eta = calculate_eta(correct_speedups)

    gamma = calculate_gamma(tolerance, pi_value4errno, errno2tolerance, b)

    return {
        "alpha": alpha,
        "beta": beta,
        "lambda": lambda_,
        "eta": eta,
        "gamma": gamma,
        "pi": pi,
    }
