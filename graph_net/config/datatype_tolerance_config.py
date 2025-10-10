def get_precision(t, dtype):
    dtype_map = {
        "float16": (3 / 5, 1),
        "bfloat16": (1.796 / 5, 1),
        "float32": (5.886 / 5, 1),
        "float64": (7 / 5, 7 / 5),
    }

    for key, (exp1, exp2) in dtype_map.items():
        if dtype == key:
            return 10 ** (t * exp1), 10 ** (t * exp2)

    return 0, 0
