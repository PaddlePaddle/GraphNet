from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [1024],
        "L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_0_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_0_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_0_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_0_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_0_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_0_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_0_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_10_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_10_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_10_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_10_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_10_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_10_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_10_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_11_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_11_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_11_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_11_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_11_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_11_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_11_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_12_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_12_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_12_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_12_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_12_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_12_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_12_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_13_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_13_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_13_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_13_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_13_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_13_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_13_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_14_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_14_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_14_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_14_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_14_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_14_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_14_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_15_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_15_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_15_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_15_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_15_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_15_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_15_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_16_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_16_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_16_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_16_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_16_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_16_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_16_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_17_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_17_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_17_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_17_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_17_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_17_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_17_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_18_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_18_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_18_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_18_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_18_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_18_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_18_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_19_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_19_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_19_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_19_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_19_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_19_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_19_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_1_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_1_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_1_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_1_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_1_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_1_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_2_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_2_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_2_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_2_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_2_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_2_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_3_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_3_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_3_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_3_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_3_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_3_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_4_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_4_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_4_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_4_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_4_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_4_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_5_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_5_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_5_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_5_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_5_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_5_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_6_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_6_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_6_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_6_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_6_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_6_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_6_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_7_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_7_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_7_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_7_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_7_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_7_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_7_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_8_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_8_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_8_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_8_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_8_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_8_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_8_modules_3_parameters_weight_"),
    (
        [1024],
        "L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 9, 9],
        "L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_blocks_modules_9_modules_0_modules_fn_modules_2_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_9_modules_1_parameters_bias_"),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_blocks_modules_9_modules_1_parameters_weight_",
    ),
    ([1024], "L_self_modules_blocks_modules_9_modules_3_buffers_running_mean_"),
    ([1024], "L_self_modules_blocks_modules_9_modules_3_buffers_running_var_"),
    ([1024], "L_self_modules_blocks_modules_9_modules_3_parameters_bias_"),
    ([1024], "L_self_modules_blocks_modules_9_modules_3_parameters_weight_"),
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 1024], "L_self_modules_head_parameters_weight_"),
    ([1024], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([1024, 3, 14, 14], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([1024], "L_self_modules_stem_modules_2_buffers_running_mean_"),
    ([1024], "L_self_modules_stem_modules_2_buffers_running_var_"),
    ([1024], "L_self_modules_stem_modules_2_parameters_bias_"),
    ([1024], "L_self_modules_stem_modules_2_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
