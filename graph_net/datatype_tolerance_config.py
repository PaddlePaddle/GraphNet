def tolerance_generator(t, dtype):
    if "float16" in dtype:
        return 10 ** (t * 3 / 5), 10**t
    elif "bfloat16" in dtype:
        return 10 ** (t * 1.796 / 5), 10**t
    elif "float32" in dtype:
        return 10 ** (t * 5.886 / 5), 10**t
    elif "float64" in dtype:
        return 10 ** (t * 7 / 5), 10 ** (t * 7 / 5)
    else:
        return 0, 0
