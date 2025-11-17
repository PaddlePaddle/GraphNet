"""
Aggregated statistics calculation module for S(t) and ES(t) metrics.

This module provides independent functions for calculating each aggregated parameter
(alpha, beta, gamma, lambda, eta, pi) according to Appendix B and C of the paper.
"""

from scipy.stats import gmean
from collections.abc import Callable


def get_errno_from_error_type(error_type: str) -> int:
    """
    Map error type string to errno (error number) for sorting.

    According to the paper:
    - c=1: accuracy errors (精度错误)
    - c=2: runtime crashes (运行时崩溃)
    - c=3: compilation failures (编译失败)

    Args:
        error_type: Error type string (e.g., "accuracy", "eager", "compiled")

    Returns:
        Errno (1, 2, or 3) based on error type
    """
    if error_type == "accuracy":
        return 1
    elif error_type in ("eager", "other", "runtime_fail", "eager_fail"):
        return 2
    elif error_type in ("compiled", "compile_fail"):
        return 3
    else:
        # Default to 2 for unknown error types (runtime errors)
        return 2


def get_error_type_from_errno(errno: int) -> str:
    """
    Map errno (error number) back to error type string.

    This is the reverse mapping of get_errno_from_error_type.
    Used when error type string information is needed.

    Args:
        errno: Error number (1, 2, or 3)

    Returns:
        Error type string:
        - 1 -> "accuracy"
        - 2 -> "runtime_fail"
        - 3 -> "compile_fail"
    """
    errno_to_error_type = {
        1: "accuracy",
        2: "runtime_fail",
        3: "compile_fail",
    }
    return errno_to_error_type.get(errno, "runtime_fail")


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
    errno2count: dict[int, int], total_samples: int, correct_speedups: list[float]
) -> dict[int, float]:
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
    if error_count == 0:
        return {errno: 0.0 for errno in errno2count.keys()}

    pi = {}
    for errno, count in errno2count.items():
        pi[errno] = count / error_count
    return pi


def calculate_gamma(
    tolerance: int,
    get_pi: Callable[[int], float],
    errno_tolerances: list[int],
    b: float = 0.1,
) -> float:
    """
    Calculate gamma_t: average error penalty factor.

    According to Appendix C: gamma_t = b^(sum(π_c * indicator(t < threshold_c)))
    where indicator(t < threshold_c) = 1 if error type c is not tolerated at tolerance t, else 0.

    Args:
        tolerance: Tolerance level t
        get_pi: Function that takes error type index c and returns π_c (proportion of error type c)
        errno_tolerances: List of tolerance thresholds for each error type.
            Index corresponds to error type index c, value is the threshold.
            An error type is tolerated (not penalized) when t >= threshold.
        b: Base penalty for severe errors (default: 0.1)

    Returns:
        Gamma value (average error penalty)
    """
    if len(errno_tolerances) == 0:
        return b

    # Calculate indicator for each error type: 1 if not tolerated, 0 if tolerated
    pi_sum = 0.0
    for error_type_index in range(len(errno_tolerances)):
        pi_c = get_pi(error_type_index)
        threshold_c = errno_tolerances[error_type_index]
        # Error type is not tolerated (penalized) when t < threshold
        indicator = 1 if tolerance < threshold_c else 0
        pi_sum += pi_c * indicator

    return b**pi_sum


def calculate_s_t_from_aggregated(
    alpha: float,
    beta: float,
    lambda_: float,
    eta: float,
    negative_speedup_penalty: float,
    fpdb: float,
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
        fpdb: Base penalty b for severe errors

    Returns:
        S(t) value calculated from aggregated parameters
    """
    return (
        alpha**lambda_
        * beta ** (lambda_ * eta * negative_speedup_penalty)
        * fpdb ** (1 - lambda_)
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


def calculate_all_aggregated_parameters(
    total_samples: int,
    correct_speedups: list[float],
    errno2count: dict[int, int],
    t_key: int,
    negative_speedup_penalty: float = 0.0,
    fpdb: float = 0.1,
    pi: dict[int, float] | None = None,
    errno_tolerance_thresholds: dict[int, int] | None = None,
) -> dict:
    """
    Calculate all aggregated parameters for a given tolerance level.

    This is a convenience function that calculates all aggregated parameters at once.

    Args:
        total_samples: Total number of samples
        correct_speedups: List of speedup values for correct samples
        errno2count: Dictionary mapping errno (error number) to their counts.
            Errno values: 1=accuracy, 2=runtime, 3=compilation.
        t_key: Tolerance level
        negative_speedup_penalty: Penalty power p for negative speedup
        fpdb: Base penalty b for severe errors
        pi: Dictionary mapping errno to their proportions (calculated at t=1).
            If None, will be calculated from errno2count.
        errno_tolerance_thresholds: Dictionary mapping errno to their tolerance thresholds.
            An error type is tolerated (not penalized) when t >= threshold.
            If None, uses default thresholds: {1: 1} for accuracy errors (errno=1), {2: 3, 3: 3} for others.

    Returns:
        Dictionary containing all aggregated parameters and calculated scores:
        {
            'alpha': float,
            'beta': float,
            'lambda': float,
            'eta': float,
            'gamma': float,
            'pi': dict[int, float],
            's_t': float,
            'es_t': float
        }
    """
    # Use default error tolerance thresholds if not provided
    if errno_tolerance_thresholds is None:
        errno_tolerance_thresholds = {}
        for errno in errno2count.keys():
            if errno == 1:  # accuracy errors
                errno_tolerance_thresholds[errno] = 1
            else:  # runtime (2) or compilation (3) errors
                errno_tolerance_thresholds[errno] = 3

    # Calculate pi if not provided
    if pi is None:
        pi = calculate_pi(errno2count, total_samples, correct_speedups)

    # Convert dictionary-based pi and thresholds to indexed format for calculate_gamma
    # Create ordered list of errnos for consistent indexing (sorted by errno)
    errnos = sorted(errno2count.keys())
    errno_tolerances = [errno_tolerance_thresholds.get(errno, 3) for errno in errnos]

    # Create get_pi function that maps error type index to pi value
    def get_pi(error_type_index: int) -> float:
        if error_type_index < len(errnos):
            errno = errnos[error_type_index]
            return pi.get(errno, 0.0)
        return 0.0

    alpha = calculate_alpha(correct_speedups)
    beta = calculate_beta(correct_speedups)
    lambda_ = calculate_lambda(correct_speedups, total_samples)
    eta = calculate_eta(correct_speedups)
    gamma = calculate_gamma(t_key, get_pi, errno_tolerances, fpdb)

    s_t = calculate_s_t_from_aggregated(
        alpha, beta, lambda_, eta, negative_speedup_penalty, fpdb
    )
    es_t = calculate_es_t_from_aggregated(
        alpha, beta, lambda_, eta, gamma, negative_speedup_penalty
    )

    return {
        "alpha": alpha,
        "beta": beta,
        "lambda": lambda_,
        "eta": eta,
        "gamma": gamma,
        "pi": pi,
        "s_t": s_t,
        "es_t": es_t,
    }
