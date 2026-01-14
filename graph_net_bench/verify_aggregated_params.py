import argparse
from collections import OrderedDict, Counter
from graph_net_bench import analysis_util
from graph_net_bench import samples_statistics
from graph_net_bench.positive_tolerance_interpretation import (
    PositiveToleranceInterpretation,
)
from graph_net_bench.samples_statistics import (
    get_errno_from_error_type,
)


def determine_tolerances(
    samples: list,
    positive_tolerance_interpretation: PositiveToleranceInterpretation,
) -> range:
    """Determine tolerance range based on observed errno categories."""
    max_errno = positive_tolerance_interpretation.num_errno_enum_values()
    return range(-10, max_errno + 2)


def extract_statistics_at_tolerance(
    samples: list,
    tolerance: int,
    positive_tolerance_interpretation: PositiveToleranceInterpretation,
) -> dict:
    """Extract statistics for a given tolerance level."""
    sample_data = [
        (
            idx,
            sample,
            sample.get("performance", {}).get("speedup", {}).get("e2e"),
            *analysis_util.check_sample_correctness(sample, tolerance),
        )
        for idx, sample in enumerate(samples)
    ]

    correct_samples = [
        (idx, speedup) for idx, _, speedup, is_correct, _ in sample_data if is_correct
    ]
    correct_count = len(correct_samples)
    correct_speedups = [
        speedup for _, speedup in correct_samples if speedup is not None
    ]
    slowdown_speedups = [speedup for speedup in correct_speedups if speedup < 1]
    correct_negative_speedup_count = len(slowdown_speedups)

    errno2count = dict(
        Counter(
            get_errno_from_error_type(fail_type, positive_tolerance_interpretation)
            for _, _, _, _, fail_type in sample_data
            if fail_type is not None
        )
    )

    return {
        "correct_count": correct_count,
        "correct_speedups": correct_speedups,
        "slowdown_speedups": slowdown_speedups,
        "correct_negative_speedup_count": correct_negative_speedup_count,
        "errno2count": errno2count,
    }


def _freeze_statistics_at_tolerance(
    stats: dict,
    total_samples: int,
    frozen_stats: dict,
    first_errno_tolerance: int,
) -> dict:
    """Freeze statistics at first_errno_tolerance and calculate pi."""
    pi = samples_statistics.calculate_pi(
        stats["errno2count"], total_samples, stats["correct_speedups"]
    )
    frozen_stats.update(
        {
            "correct_count": stats["correct_count"],
            "correct_negative_speedup_count": stats["correct_negative_speedup_count"],
            "correct_speedups": stats["correct_speedups"],
            "slowdown_speedups": stats["slowdown_speedups"],
            "errno2count": stats["errno2count"].copy(),
        }
    )
    return pi


def select_statistics_for_calculation(
    tolerance: int, current_stats: dict, frozen_stats: dict
) -> dict:
    """Select statistics to use based on tolerance level."""
    if tolerance < 1:
        return {
            "correct_count": current_stats["correct_count"],
            "correct_speedups": current_stats["correct_speedups"],
            "slowdown_speedups": current_stats["slowdown_speedups"],
            "errno2count": current_stats["errno2count"],
        }
    else:
        return {
            "correct_count": frozen_stats["correct_count"],
            "correct_speedups": frozen_stats["correct_speedups"],
            "slowdown_speedups": frozen_stats["slowdown_speedups"],
            "errno2count": frozen_stats["errno2count"],
        }


def calculate_es_constructor_params_for_tolerance(
    tolerance: int,
    total_samples: int,
    stats: dict,
    pi: dict,
    negative_speedup_penalty: float,
    fpdb: float,
    positive_tolerance_interpretation: PositiveToleranceInterpretation,
) -> dict:
    """Calculate ES(t) constructor parameters (alpha, beta, gamma, lambda, eta) and final scores for a tolerance level."""
    aggregated_params = samples_statistics.calculate_es_components_values(
        total_samples=total_samples,
        correct_speedups=stats["correct_speedups"],
        errno2count=stats["errno2count"],
        tolerance=tolerance,
        negative_speedup_penalty=negative_speedup_penalty,
        b=fpdb,
        pi=pi,
        positive_tolerance_interpretation=positive_tolerance_interpretation,
    )

    alpha = aggregated_params["alpha"]
    beta = aggregated_params["beta"]
    lambda_ = aggregated_params["lambda"]
    eta = aggregated_params["eta"]
    gamma = aggregated_params["gamma"]

    expected_s = samples_statistics.calculate_s_t_from_aggregated(
        alpha, beta, lambda_, eta, negative_speedup_penalty, fpdb
    )
    expected_es = samples_statistics.calculate_es_t_from_aggregated(
        alpha, beta, lambda_, eta, gamma, negative_speedup_penalty
    )

    return {
        "alpha": alpha,
        "beta": beta,
        "lambda": lambda_,
        "eta": eta,
        "gamma": gamma,
        "expected_s": expected_s,
        "expected_es": expected_es,
    }


def print_tolerance_details(
    tolerance: int,
    total_samples: int,
    stats: dict,
    es_constructor_params: dict,
    pi: dict,
    fpdb: float,
):
    """Print detailed information for a tolerance level."""
    print(f"\nTolerance t = {tolerance}:")
    print(f"  Total samples: {total_samples}")
    print(
        f"  Correct samples: {stats['correct_count']} (lambda = {es_constructor_params['lambda']:.6f})"
    )
    print(f"  Correct speedups collected: {len(stats['correct_speedups'])}")
    print(
        f"  Slowdown cases: {len(stats['slowdown_speedups'])} (eta = {es_constructor_params['eta']:.6f})"
    )
    print(
        f"  alpha (geometric mean of correct speedups): {es_constructor_params['alpha']:.6f}"
    )
    if stats["correct_speedups"]:
        print(
            f"    - Correct speedups: {stats['correct_speedups'][:10]}{'...' if len(stats['correct_speedups']) > 10 else ''}"
        )
    print(
        f"  beta (geometric mean of slowdown speedups): {es_constructor_params['beta']:.6f}"
    )
    if stats["slowdown_speedups"]:
        print(
            f"    - Slowdown speedups: {stats['slowdown_speedups'][:10]}{'...' if len(stats['slowdown_speedups']) > 10 else ''}"
        )
    print(f"  gamma (average error penalty): {es_constructor_params['gamma']:.6f}")
    if tolerance >= 1:
        errnos = sorted(pi.keys())
        pi_list = [pi[errno] for errno in errnos]
        indicator = [1 if tolerance < 1 else 0, 1 if tolerance < 3 else 0]
        pi_indicator_sum = sum(
            pi.get(errno, 0.0) * indicator[min(i, len(indicator) - 1)]
            for i, errno in enumerate(errnos)
        )
        print(f"    - pi (errno -> proportion): {dict(sorted(pi.items()))}")
        print(f"    - pi (as list): {pi_list}")
        print(f"    - indicator: {indicator}")
        print(
            f"    - gamma = fpdb^(sum(pi[i] * indicator[i])) = {fpdb}^{pi_indicator_sum:.6f} = {es_constructor_params['gamma']:.6f}"
        )
    print(f"  Expected S(t) from aggregated: {es_constructor_params['expected_s']:.6f}")
    print(
        f"  Expected ES(t) from aggregated: {es_constructor_params['expected_es']:.6f}"
    )


class ToleranceReportBuilder:
    """Stateful helper for building tolerance reports in order."""

    def __init__(
        self,
        samples: list,
        total_samples: int,
        negative_speedup_penalty: float,
        fpdb: float,
        positive_tolerance_interpretation: PositiveToleranceInterpretation,
    ):
        self.samples = samples
        self.total_samples = total_samples
        self.negative_speedup_penalty = negative_speedup_penalty
        self.fpdb = fpdb
        self.pi: dict[int, float] = {}
        self.frozen_stats: dict = {
            "correct_count": 0,
            "correct_speedups": [],
            "slowdown_speedups": [],
            "errno2count": {},
        }
        self.positive_tolerance_interpretation = positive_tolerance_interpretation

    def build_report(self, tolerance: int) -> dict:
        current_stats = extract_statistics_at_tolerance(
            self.samples, tolerance, self.positive_tolerance_interpretation
        )

        if tolerance == 1:
            self.pi = _freeze_statistics_at_tolerance(
                current_stats,
                self.total_samples,
                self.frozen_stats,
                first_errno_tolerance=1,
            )

        if self.total_samples == 0:
            return self._empty_report(tolerance)

        stats_for_calc = select_statistics_for_calculation(
            tolerance, current_stats, self.frozen_stats
        )
        # For tolerance < 1, pass None to let calculate_es_components_values recalculate pi
        # For tolerance >= 1, use frozen pi from t=1
        pi_for_calc = None if tolerance < 1 else self.pi
        es_constructor_params = calculate_es_constructor_params_for_tolerance(
            tolerance,
            self.total_samples,
            stats_for_calc,
            pi_for_calc,
            self.negative_speedup_penalty,
            self.fpdb,
            self.positive_tolerance_interpretation,
        )
        # Use calculated pi from es_constructor_params for display and return
        calculated_pi = es_constructor_params.get("pi", self.pi)
        print_tolerance_details(
            tolerance,
            self.total_samples,
            stats_for_calc,
            es_constructor_params,
            calculated_pi,
            self.fpdb,
        )
        return {
            **es_constructor_params,
            "pi": calculated_pi,
            "correct_count": stats_for_calc["correct_count"],
            "total_samples": self.total_samples,
            "correct_speedups_count": len(stats_for_calc["correct_speedups"]),
            "slowdown_count": len(stats_for_calc["slowdown_speedups"]),
        }

    def _empty_report(self, tolerance: int) -> dict:
        print(f"\nTolerance t = {tolerance}: No samples to analyze")
        return {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": self.fpdb,
            "lambda": 0.0,
            "eta": 0.0,
            "pi": self.pi,
            "expected_s": self.fpdb,
            "expected_es": self.fpdb,
            "correct_count": 0,
            "total_samples": 0,
            "correct_speedups_count": 0,
            "slowdown_count": 0,
        }


def verify_es_constructor_params_across_tolerances(
    samples: list,
    folder_name: str,
    positive_tolerance_interpretation: PositiveToleranceInterpretation,
    negative_speedup_penalty: float = 0,
    fpdb: float = 0.1,
) -> dict:
    """
    Verify and print ES constructor parameters (alpha, beta, gamma, lambda, eta, pi) for each
    tolerance level independently. This logic mirrors `calculate_s_scores` but is split
    out for focused validation of aggregated calculations.

    Returns:
        Dictionary mapping tolerance -> dict of ES constructor parameters and calculated scores
    """
    total_samples = len(samples)

    print(f"\n{'=' * 80}")
    print(f"Verifying Aggregated Parameters for '{folder_name}'")
    print(f"{'=' * 80}")

    tolerances = determine_tolerances(samples, positive_tolerance_interpretation)
    builder = ToleranceReportBuilder(
        samples=samples,
        total_samples=total_samples,
        negative_speedup_penalty=negative_speedup_penalty,
        fpdb=fpdb,
        positive_tolerance_interpretation=positive_tolerance_interpretation,
    )

    results = OrderedDict(
        (tolerance, builder.build_report(tolerance)) for tolerance in tolerances
    )

    print(f"\n{'=' * 80}")
    print("Aggregated Parameter Verification Complete")
    print(f"{'=' * 80}\n")

    return results


def main():
    """Main execution function for verifying aggregated parameters."""
    parser = argparse.ArgumentParser(
        description="Verify aggregated parameters (alpha, beta, gamma, lambda, eta, pi) calculation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="Path to the benchmark log file or directory containing benchmark JSON files or sub-folders.",
    )
    parser.add_argument(
        "--negative-speedup-penalty",
        type=float,
        default=0.0,
        help="Penalty power (p) for negative speedup. Formula: speedup**(p+1). Default: 0.0.",
    )
    parser.add_argument(
        "--fpdb",
        type=float,
        default=0.1,
        help="Base penalty for severe errors (e.g., crashes, correctness failures).",
    )
    args = parser.parse_args()

    # Scan folders to get data
    all_results = analysis_util.scan_all_folders(args.benchmark_path)
    if not all_results:
        print("No valid data found. Exiting.")
        return

    # Calculate and print aggregated parameters for each curve
    for folder_name, samples in all_results.items():
        _ = verify_es_constructor_params_across_tolerances(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            fpdb=args.fpdb,
        )  # noqa: F841


if __name__ == "__main__":
    main()
