import os


def tolerance_generator(t):
    # for float16
    yield 10 ** (t * 3 / 5), 10**t
    # for bfloat16
    yield 10 ** (t * 1.796 / 5), 10**t
    # yield float32
    yield 10 ** (t * 5.886 / 5), 10**t
    # yield float64
    yield 10 ** (t * 7 / 5), 10 ** (t * 7 / 5)


def calculate_tolerance_pair(begin, end):
    tolerance_pair_list = []
    for t in range(begin, end + 1):
        for rtol, atol in tolerance_generator(t):
            effective_atol = float(f"{atol:.3g}")
            effective_rtol = float(f"{rtol:.3g}")
            tolerance_pair_list.append(
                {
                    "atol": effective_atol,
                    "rtol": effective_rtol,
                }
            )
    return tolerance_pair_list


def generate_allclose_configs(cmp_all_close_func):
    tolerance_pair_list = calculate_tolerance_pair(-10, 5)

    cmp_configs = []
    for pair in tolerance_pair_list:
        atol, rtol = pair["atol"], pair["rtol"]
        cmp_configs.append(
            (f"[all_close_atol_{atol:.2E}_rtol_{rtol:.2E}]", cmp_all_close_func, pair)
        )
    return cmp_configs
