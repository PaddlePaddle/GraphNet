import os
import argparse
import numpy as np
from collections import OrderedDict
from graph_net import analysis_util
from graph_net import macro_statistics


def calculate_macro_parameters(
    samples: list,
    folder_name: str,
    negative_speedup_penalty: float = 0,
    fpdb: float = 0.1,
) -> dict:
    """
    Calculate and print all macro parameters (alpha, beta, gamma, lambda, eta, pi)
    for each tolerance level independently.

    This function extracts the macro parameter calculation logic from calculate_s_scores
    to verify the correctness of macro-level calculations.

    Returns:
        Dictionary mapping tolerance -> dict of macro parameters and calculated scores
    """
    begin = -10
    end = 4
    t_keys = list(range(begin, end + 1))
    total_samples = len(samples)

    print(f"\n{'='*80}")
    print(f"Verifying Macro Parameters for '{folder_name}'")
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

    results = OrderedDict()

    for t_key in t_keys:
        correct_count = 0
        acc_failure_count = 0
        correct_negative_speedup_count = 0
        correct_speedups = []
        slowdown_speedups = []

        # Collect statistics for current tolerance using helper function
        for idx, sample in enumerate(samples):
            performance_data = sample.get("performance", {})
            speedup = performance_data.get("speedup", {}).get("e2e")

            # Check correctness using dedicated function
            is_correct, fail_type = analysis_util.check_sample_correctness(
                sample, t_key
            )

            # Collect statistics
            if is_correct:
                correct_count += 1
                if speedup is not None:
                    correct_speedups.append(speedup)
                if speedup is not None and speedup < 1:
                    correct_negative_speedup_count += 1
                    slowdown_speedups.append(speedup)

            if fail_type == "accuracy":
                acc_failure_count += 1

            # Store state at t=1
            if t_key == 1:
                is_correct_at_t1[idx] = is_correct
                speedup_at_t1[idx] = speedup
                fail_type_at_t1[idx] = fail_type if fail_type is not None else "CORRECT"

        # Calculate pi at t=1 using the dedicated function
        if t_key == 1:
            pi = macro_statistics.calculate_pi(
                acc_failure_count, total_samples, correct_count
            )
            final_correct_count = correct_count
            final_correct_negative_speedup_count = correct_negative_speedup_count
            final_correct_speedups = correct_speedups
            final_slowdown_speedups = slowdown_speedups

        # Calculate macro parameters
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

            # Calculate all macro parameters using the dedicated module
            macro_params = macro_statistics.calculate_all_macro_parameters(
                correct_count=stats_correct_count,
                total_samples=total_samples,
                correct_negative_speedup_count=stats_correct_negative_speedup_count,
                correct_speedups=stats_correct_speedups,
                slowdown_speedups=stats_slowdown_speedups,
                acc_failure_count=acc_failure_count,
                t_key=t_key,
                negative_speedup_penalty=negative_speedup_penalty,
                fpdb=fpdb,
                pi=pi,
            )

            alpha = macro_params["alpha"]
            beta = macro_params["beta"]
            lambda_ = macro_params["lambda"]
            eta = macro_params["eta"]
            gamma = macro_params["gamma"]
            expected_s = macro_params["s_t"]
            expected_es = macro_params["es_t"]

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
                indicator = [1 if t_key < 1 else 0, 1 if t_key < 3 else 0]
                pi_indicator_sum = sum(pi[i] * indicator[i] for i in range(len(pi)))
                print(f"    - pi: {list(pi)}")
                print(f"    - indicator: {indicator}")
                print(
                    f"    - gamma = fpdb^(sum(pi[i] * indicator[i])) = {fpdb}^{pi_indicator_sum:.6f} = {gamma:.6f}"
                )
            print(f"  Expected S(t) from macro: {expected_s:.6f}")
            print(f"  Expected ES(t) from macro: {expected_es:.6f}")
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
    print(f"Macro Parameter Verification Complete")
    print(f"{'='*80}\n")

    return results


def main():
    """Main execution function for verifying macro parameters."""
    parser = argparse.ArgumentParser(
        description="Verify macro parameters (alpha, beta, gamma, lambda, eta, pi) calculation.",
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

    # Calculate and print macro parameters for each curve
    for folder_name, samples in all_results.items():
        macro_results = calculate_macro_parameters(
            samples,
            folder_name,
            negative_speedup_penalty=args.negative_speedup_penalty,
            fpdb=args.fpdb,
        )


if __name__ == "__main__":
    main()
