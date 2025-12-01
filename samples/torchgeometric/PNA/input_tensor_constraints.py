from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 2}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 100], "L_edge_index_"),
    ([1], "L_self_modules_convs_modules_0_modules_aggr_module_buffers_avg_deg_log_"),
    ([256], "L_self_modules_convs_modules_0_modules_lin_parameters_bias_"),
    ([256, 256], "L_self_modules_convs_modules_0_modules_lin_parameters_weight_"),
    (
        [256],
        "L_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 1664],
        "L_self_modules_convs_modules_0_modules_post_nns_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_convs_modules_0_modules_pre_nns_modules_0_modules_0_parameters_weight_",
    ),
    ([1], "L_self_modules_convs_modules_1_modules_aggr_module_buffers_avg_deg_log_"),
    ([256], "L_self_modules_convs_modules_1_modules_lin_parameters_bias_"),
    ([256, 256], "L_self_modules_convs_modules_1_modules_lin_parameters_weight_"),
    (
        [256],
        "L_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 3328],
        "L_self_modules_convs_modules_1_modules_post_nns_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 512],
        "L_self_modules_convs_modules_1_modules_pre_nns_modules_0_modules_0_parameters_weight_",
    ),
    ([1], "L_self_modules_convs_modules_2_modules_aggr_module_buffers_avg_deg_log_"),
    ([256], "L_self_modules_convs_modules_2_modules_lin_parameters_bias_"),
    ([256, 256], "L_self_modules_convs_modules_2_modules_lin_parameters_weight_"),
    (
        [256],
        "L_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 3328],
        "L_self_modules_convs_modules_2_modules_post_nns_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 512],
        "L_self_modules_convs_modules_2_modules_pre_nns_modules_0_modules_0_parameters_weight_",
    ),
    ([1], "L_self_modules_convs_modules_3_modules_aggr_module_buffers_avg_deg_log_"),
    ([256], "L_self_modules_convs_modules_3_modules_lin_parameters_bias_"),
    ([256, 256], "L_self_modules_convs_modules_3_modules_lin_parameters_weight_"),
    (
        [256],
        "L_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 3328],
        "L_self_modules_convs_modules_3_modules_post_nns_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 512],
        "L_self_modules_convs_modules_3_modules_pre_nns_modules_0_modules_0_parameters_weight_",
    ),
    ([1], "L_self_modules_convs_modules_4_modules_aggr_module_buffers_avg_deg_log_"),
    ([10], "L_self_modules_convs_modules_4_modules_lin_parameters_bias_"),
    ([10, 10], "L_self_modules_convs_modules_4_modules_lin_parameters_weight_"),
    (
        [10],
        "L_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [10, 3328],
        "L_self_modules_convs_modules_4_modules_post_nns_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256, 512],
        "L_self_modules_convs_modules_4_modules_pre_nns_modules_0_modules_0_parameters_weight_",
    ),
    ([1000, 128], "L_x_"),
]
