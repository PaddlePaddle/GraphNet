import os
import argparse
import numpy as np
from collections import OrderedDict, Counter
from graph_net import analysis_util
from graph_net import samples_statistics
from graph_net.samples_statistics import (
    get_errno_from_error_type,
    get_error_type_from_errno,
)


def calculate_aggregated_parameters(
    samples: list,
    folder_name: str,
    negative_speedup_penalty: float = 0,
    fpdb: float = 0.1,
) -> dict:
    """
    Calculate and print all aggregated parameters (alpha, beta, gamma, lambda, eta, pi)
    for each tolerance level independently.

    This function extracts the aggregated parameter calculation logic from calculate_s_scores
    to verify the correctness of aggregated-level calculations.

    Returns:
        Dictionary mapping tolerance -> dict of aggregated parameters and calculated scores
    """
    begin = -10
    end = 4
    t_keys = list(range(begin, end + 1))
    total_samples = len(samples)

    print(f"\n{'='*80}")
    print(f"Verifying Aggregated Parameters for '{folder_name}'")
    print(f"{'='*80}")

    # pi is a tuple of constants for t > 0 for each group: (pi[0], pi[1])
    # Calculated at t=1, used for all t >= 1
    pi = (0.0, 0.0)

    # Store state at t=1 for ES(t) calculation
    is_correct_at_t1 = [False] * total_samples
    speedup_at_t1 = [None] * total_samples
    fail_type_at_t1 = ["CORRECT"] * total_samples

    # Final statistics at t=1
    final_correct_count = 0
    final_correct_negative_speedup_count = 0
    final_correct_speedups = []
    final_slowdown_speedups = []
    final_errno2count = {}  # Store error type counts at t=1 (using errno)

    results = OrderedDict()

    for t_key in t_keys:
        # Extract sample data using map
        sample_data = [
            (
                idx,
                sample,
                sample.get("performance", {}).get("speedup", {}).get("e2e"),
                *analysis_util.check_sample_correctness(sample, t_key),
            )
            for idx, sample in enumerate(samples)
        ]

        # Filter correct samples and extract speedups
        correct_samples = [
            (idx, speedup)
            for idx, _, speedup, is_correct, _ in sample_data
            if is_correct
        ]
        correct_count = len(correct_samples)
        correct_speedups = [
            speedup for _, speedup in correct_samples if speedup is not None
        ]
        slowdown_speedups = [speedup for speedup in correct_speedups if speedup < 1]
        correct_negative_speedup_count = len(slowdown_speedups)

        # Count errors by errno using Counter (convert error type string to errno)
        errno2count = dict(
            Counter(
                get_errno_from_error_type(fail_type)
                for _, _, _, _, fail_type in sample_data
                if fail_type is not None
            )
        )

        # Store state at t=1 using list comprehension
        if t_key == 1:
            t1_data = [
                (
                    idx,
                    speedup,
                    is_correct,
                    fail_type if fail_type is not None else "CORRECT",
                )
                for idx, _, speedup, is_correct, fail_type in sample_data
            ]
            is_correct_at_t1 = [is_correct for _, _, is_correct, _ in t1_data]
            speedup_at_t1 = [speedup for _, speedup, _, _ in t1_data]
            fail_type_at_t1 = [fail_type for _, _, _, fail_type in t1_data]

        # Calculate pi at t=1 using the dedicated function
        if t_key == 1:
            pi = samples_statistics.calculate_pi(
                errno2count, total_samples, correct_speedups
            )
            final_correct_count = correct_count
            final_correct_negative_speedup_count = correct_negative_speedup_count
            final_correct_speedups = correct_speedups
            final_slowdown_speedups = slowdown_speedups
            final_errno2count = errno2count.copy()  # Save for t >= 1

        # Calculate aggregated parameters
        if total_samples > 0:
            # For t < 1, use current tolerance statistics
            # For t >= 1, use t=1 statistics (frozen state)
            if t_key < 1:
                stats_correct_count = correct_count
                stats_correct_negative_speedup_count = correct_negative_speedup_count
                stats_correct_speedups = correct_speedups
                stats_slowdown_speedups = slowdown_speedups
            else:
                stats_correct_count = final_correct_count
                stats_correct_negative_speedup_count = (
                    final_correct_negative_speedup_count
                )
                stats_correct_speedups = final_correct_speedups
                stats_slowdown_speedups = final_slowdown_speedups

            # Calculate all aggregated parameters using the dedicated module
            # For t >= 1, use errno2count from t=1 (frozen state)
            if t_key < 1:
                stats_errno2count = errno2count
            else:
                stats_errno2count = final_errno2count  # Use frozen from t=1

            aggregated_params = samples_statistics.calculate_all_aggregated_parameters(
                total_samples=total_samples,
                correct_speedups=stats_correct_speedups,
                errno2count=stats_errno2count,
                t_key=t_key,
                negative_speedup_penalty=negative_speedup_penalty,
                fpdb=fpdb,
                pi=pi,
            )

            alpha = aggregated_params["alpha"]
            beta = aggregated_params["beta"]
            lambda_ = aggregated_params["lambda"]
            eta = aggregated_params["eta"]
            gamma = aggregated_params["gamma"]
            expected_s = aggregated_params["s_t"]
            expected_es = aggregated_params["es_t"]

            results[t_key] = {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda": lambda_,
                "eta": eta,
                "pi": pi,
                "expected_s": expected_s,
                "expected_es": expected_es,
                "correct_count": stats_correct_count,
                "total_samples": total_samples,
                "correct_speedups_count": len(stats_correct_speedups),
                "slowdown_count": len(stats_slowdown_speedups),
            }

            # Print detailed information
            print(f"\nTolerance t = {t_key}:")
            print(f"  Total samples: {total_samples}")
            print(f"  Correct samples: {stats_correct_count} (lambda = {lambda_:.6f})")
            print(f"  Correct speedups collected: {len(stats_correct_speedups)}")
            print(f"  Slowdown cases: {len(stats_slowdown_speedups)} (eta = {eta:.6f})")
            print(f"  alpha (geometric mean of correct speedups): {alpha:.6f}")
            if stats_correct_speedups:
                print(
                    f"    - Correct speedups: {stats_correct_speedups[:10]}{'...' if len(stats_correct_speedups) > 10 else ''}"
                )
            print(f"  beta (geometric mean of slowdown speedups): {beta:.6f}")
            if stats_slowdown_speedups:
                print(
                    f"    - Slowdown speedups: {stats_slowdown_speedups[:10]}{'...' if len(stats_slowdown_speedups) > 10 else ''}"
                )
            print(f"  gamma (average error penalty): {gamma:.6f}")
            if t_key >= 1:
                # pi is now dict[int, float], convert to list for display
                errnos = sorted(pi.keys())
                pi_list = [pi[errno] for errno in errnos]
                indicator = [1 if t_key < 1 else 0, 1 if t_key < 3 else 0]
                # Calculate pi_indicator_sum using errno-based pi
                pi_indicator_sum = sum(
                    pi.get(errno, 0.0) * indicator[min(i, len(indicator) - 1)]
                    for i, errno in enumerate(errnos)
                )
                print(f"    - pi (errno -> proportion): {dict(sorted(pi.items()))}")
                print(f"    - pi (as list): {pi_list}")
                print(f"    - indicator: {indicator}")
                print(
                    f"    - gamma = fpdb^(sum(pi[i] * indicator[i])) = {fpdb}^{pi_indicator_sum:.6f} = {gamma:.6f}"
                )
            print(f"  Expected S(t) from aggregated: {expected_s:.6f}")
            print(f"  Expected ES(t) from aggregated: {expected_es:.6f}")
        else:
            results[t_key] = {
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": fpdb,
                "lambda": 0.0,
                "eta": 0.0,
                "pi": pi,
                "expected_s": fpdb,
                "expected_es": fpdb,
                "correct_count": 0,
                "total_samples": 0,
                "correct_speedups_count": 0,
                "slowdown_count": 0,
            }
            print(f"\nTolerance t = {t_key}: No samples to analyze")

    print(f"\n{'='*80}")
    print(f"Aggregated Parameter Verification Complete")
    print(f"{'='*80}\n")

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
        aggregated_results = calculate_aggregated_parameters(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            fpdb=args.fpdb,
        )


if __name__ == "__main__":
    main()
