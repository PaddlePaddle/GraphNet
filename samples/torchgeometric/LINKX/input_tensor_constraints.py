from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 2, S1: 100}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_edge_index_"),
    ([128], "L_self_modules_cat_lin1_parameters_bias_"),
    ([128, 128], "L_self_modules_cat_lin1_parameters_weight_"),
    ([128], "L_self_modules_cat_lin2_parameters_bias_"),
    ([128, 128], "L_self_modules_cat_lin2_parameters_weight_"),
    ([128], "L_self_modules_edge_lin_parameters_bias_"),
    ([1000, 128], "L_self_modules_edge_lin_parameters_weight_"),
    ([128], "L_self_modules_final_mlp_modules_lins_modules_0_parameters_bias_"),
    ([128, 128], "L_self_modules_final_mlp_modules_lins_modules_0_parameters_weight_"),
    ([128], "L_self_modules_final_mlp_modules_lins_modules_1_parameters_bias_"),
    ([128, 128], "L_self_modules_final_mlp_modules_lins_modules_1_parameters_weight_"),
    ([128], "L_self_modules_final_mlp_modules_lins_modules_2_parameters_bias_"),
    ([128, 128], "L_self_modules_final_mlp_modules_lins_modules_2_parameters_weight_"),
    ([128], "L_self_modules_final_mlp_modules_lins_modules_3_parameters_bias_"),
    ([128, 128], "L_self_modules_final_mlp_modules_lins_modules_3_parameters_weight_"),
    ([10], "L_self_modules_final_mlp_modules_lins_modules_4_parameters_bias_"),
    ([10, 128], "L_self_modules_final_mlp_modules_lins_modules_4_parameters_weight_"),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_0_modules_module_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_0_modules_module_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_1_modules_module_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_1_modules_module_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_2_modules_module_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_2_modules_module_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_3_modules_module_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_final_mlp_modules_norms_modules_3_modules_module_parameters_weight_",
    ),
    ([128], "L_self_modules_node_mlp_modules_lins_modules_0_parameters_bias_"),
    ([128, 128], "L_self_modules_node_mlp_modules_lins_modules_0_parameters_weight_"),
    ([1000, 128], "L_x_"),
]
