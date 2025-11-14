"""
Macro-level statistics calculation module for S(t) and ES(t) metrics.

This module provides independent functions for calculating each macro parameter
(alpha, beta, gamma, lambda, eta, pi) according to Appendix B and C of the paper.
"""

from scipy.stats import gmean
from typing import List, Tuple


def calculate_alpha(correct_speedups: List[float]) -> float:
    """
    Calculate alpha: geometric mean of correct sample speedups.

    According to Appendix B: alpha is the geometric mean of all correct sample speedups.

    Args:
        correct_speedups: List of speedup values for correct samples

    Returns:
        Alpha value (geometric mean), or 1.0 if list is empty
    """
    return gmean(correct_speedups) if correct_speedups else 1.0


def calculate_beta(slowdown_speedups: List[float]) -> float:
    """
    Calculate beta: geometric mean of slowdown sample speedups.

    According to Appendix B: beta is the geometric mean of speedups for samples
    that are correct but have speedup < 1 (slowdown cases).

    Args:
        slowdown_speedups: List of speedup values for slowdown cases (speedup < 1)

    Returns:
        Beta value (geometric mean), or 1.0 if list is empty
    """
    return gmean(slowdown_speedups) if slowdown_speedups else 1.0


def calculate_lambda(correct_count: int, total_samples: int) -> float:
    """
    Calculate lambda: fraction of correct samples.

    According to Appendix B: lambda = M / N, where M is correct count and N is total samples.

    Args:
        correct_count: Number of correct samples
        total_samples: Total number of samples

    Returns:
        Lambda value (fraction of correct samples), or 0.0 if total_samples is 0
    """
    return correct_count / total_samples if total_samples > 0 else 0.0


def calculate_eta(correct_negative_speedup_count: int, correct_count: int) -> float:
    """
    Calculate eta: fraction of slowdown cases within correct samples.

    According to Appendix B: eta = K / M, where K is the number of slowdown cases
    within correct samples, and M is the total correct count.

    Args:
        correct_negative_speedup_count: Number of correct samples with speedup < 1
        correct_count: Total number of correct samples

    Returns:
        Eta value (fraction of slowdown cases), or 0.0 if correct_count is 0
    """
    return correct_negative_speedup_count / correct_count if correct_count > 0 else 0.0


def calculate_pi(
    acc_failure_count: int, total_samples: int, correct_count: int
) -> Tuple[float, float]:
    """
    Calculate pi: error type proportions for t > 0.

    According to Appendix C: pi[0] is the proportion of accuracy errors (c=1),
    pi[1] is the proportion of other errors (c=2,3) among all error samples.

    Args:
        acc_failure_count: Number of accuracy failure samples
        total_samples: Total number of samples
        correct_count: Number of correct samples

    Returns:
        Tuple of (pi[0], pi[1]) where:
        - pi[0]: proportion of accuracy errors among error samples
        - pi[1]: proportion of other errors among error samples (1 - pi[0])
    """
    error_count = total_samples - correct_count
    if error_count == 0:
        return 0.0, 0.0

    pi_0 = acc_failure_count / error_count
    pi_1 = 1.0 - pi_0
    return pi_0, pi_1


def calculate_gamma(t_key: int, pi: Tuple[float, float], fpdb: float = 0.1) -> float:
    """
    Calculate gamma_t: average error penalty factor.

    According to Appendix C: gamma_t = b^(sum(π_c * indicator(t < c)))
    where:
    - pi[0]: proportion of accuracy errors (c=1, tolerated when t >= 1)
    - pi[1]: proportion of other errors (c=2,3, tolerated when t >= 3)
    - indicator[0] = 1 if t < 1 else 0 (accuracy errors not tolerated)
    - indicator[1] = 1 if t < 3 else 0 (runtime/compile errors not tolerated)

    Args:
        t_key: Tolerance level
        pi: Tuple of (pi[0], pi[1]) error type proportions
        fpdb: Base penalty for severe errors (default: 0.1)

    Returns:
        Gamma value (average error penalty)
    """
    if t_key < 1:
        return fpdb

    # indicator[0] = 1 if t < 1 else 0 (accuracy errors not tolerated)
    # indicator[1] = 1 if t < 3 else 0 (runtime/compile errors not tolerated)
    indicator = [1 if t_key < 1 else 0, 1 if t_key < 3 else 0]
    pi_sum = sum(pi[i] * indicator[i] for i in range(len(pi)))
    return fpdb**pi_sum


def calculate_s_t_macro(
    alpha: float,
    beta: float,
    lambda_: float,
    eta: float,
    negative_speedup_penalty: float,
    fpdb: float,
) -> float:
    """
    Calculate S(t) from macro parameters.

    According to Appendix B: S_t = α^λ · β^(ληp) · b^(1-λ)

    Args:
        alpha: Geometric mean speedup of correct samples
        beta: Geometric mean speedup of slowdown cases
        lambda_: Fraction of correct samples
        eta: Fraction of slowdown cases within correct samples
        negative_speedup_penalty: Penalty power p for negative speedup
        fpdb: Base penalty b for severe errors

    Returns:
        S(t) value calculated from macro parameters
    """
    return (
        alpha**lambda_
        * beta ** (lambda_ * eta * negative_speedup_penalty)
        * fpdb ** (1 - lambda_)
    )


def calculate_es_t_macro(
    alpha: float,
    beta: float,
    lambda_: float,
    eta: float,
    gamma: float,
    negative_speedup_penalty: float,
) -> float:
    """
    Calculate ES(t) from macro parameters.

    According to Appendix C: ES_t = α^λ · β^(ληp) · γ_t^(1-λ)

    Args:
        alpha: Geometric mean speedup of correct samples
        beta: Geometric mean speedup of slowdown cases
        lambda_: Fraction of correct samples
        eta: Fraction of slowdown cases within correct samples
        gamma: Average error penalty factor
        negative_speedup_penalty: Penalty power p for negative speedup

    Returns:
        ES(t) value calculated from macro parameters
    """
    return (
        alpha**lambda_
        * beta ** (lambda_ * eta * negative_speedup_penalty)
        * gamma ** (1 - lambda_)
    )


def calculate_all_macro_parameters(
    correct_count: int,
    total_samples: int,
    correct_negative_speedup_count: int,
    correct_speedups: List[float],
    slowdown_speedups: List[float],
    acc_failure_count: int,
    t_key: int,
    negative_speedup_penalty: float = 0.0,
    fpdb: float = 0.1,
    pi: Tuple[float, float] = (0.0, 0.0),
) -> dict:
    """
    Calculate all macro parameters for a given tolerance level.

    This is a convenience function that calculates all macro parameters at once.

    Args:
        correct_count: Number of correct samples
        total_samples: Total number of samples
        correct_negative_speedup_count: Number of slowdown cases
        correct_speedups: List of speedup values for correct samples
        slowdown_speedups: List of speedup values for slowdown cases
        acc_failure_count: Number of accuracy failure samples
        t_key: Tolerance level
        negative_speedup_penalty: Penalty power p for negative speedup
        fpdb: Base penalty b for severe errors
        pi: Tuple of (pi[0], pi[1]) error type proportions (calculated at t=1)

    Returns:
        Dictionary containing all macro parameters and calculated scores:
        {
            'alpha': float,
            'beta': float,
            'lambda': float,
            'eta': float,
            'gamma': float,
            'pi': Tuple[float, float],
            's_t': float,
            'es_t': float
        }
    """
    alpha = calculate_alpha(correct_speedups)
    beta = calculate_beta(slowdown_speedups)
    lambda_ = calculate_lambda(correct_count, total_samples)
    eta = calculate_eta(correct_negative_speedup_count, correct_count)
    gamma = calculate_gamma(t_key, pi, fpdb)

    s_t = calculate_s_t_macro(alpha, beta, lambda_, eta, negative_speedup_penalty, fpdb)
    es_t = calculate_es_t_macro(
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
