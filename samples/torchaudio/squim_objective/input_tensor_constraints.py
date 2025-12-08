from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [1024],
        "L_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_branches_modules_0_modules_0_modules_linear1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_branches_modules_0_modules_0_modules_linear2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_branches_modules_0_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_branches_modules_0_modules_0_modules_self_attn_parameters_in_proj_weight_",
    ),
    ([1], "L_self_modules_branches_modules_0_modules_1_parameters_alpha_"),
    ([256], "L_self_modules_branches_modules_0_modules_2_modules_0_parameters_bias_"),
    (
        [256, 256],
        "L_self_modules_branches_modules_0_modules_2_modules_0_parameters_weight_",
    ),
    ([1], "L_self_modules_branches_modules_0_modules_2_modules_1_parameters_weight_"),
    ([1], "L_self_modules_branches_modules_0_modules_2_modules_2_parameters_bias_"),
    (
        [1, 256],
        "L_self_modules_branches_modules_0_modules_2_modules_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_branches_modules_1_modules_0_modules_linear1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_branches_modules_1_modules_0_modules_linear2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_branches_modules_1_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_branches_modules_1_modules_0_modules_self_attn_parameters_in_proj_weight_",
    ),
    ([1], "L_self_modules_branches_modules_1_modules_1_parameters_alpha_"),
    ([256], "L_self_modules_branches_modules_1_modules_2_modules_0_parameters_bias_"),
    (
        [256, 256],
        "L_self_modules_branches_modules_1_modules_2_modules_0_parameters_weight_",
    ),
    ([1], "L_self_modules_branches_modules_1_modules_2_modules_1_parameters_weight_"),
    ([1], "L_self_modules_branches_modules_1_modules_2_modules_2_parameters_bias_"),
    (
        [1, 256],
        "L_self_modules_branches_modules_1_modules_2_modules_2_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_bias_",
    ),
    (
        [1024, 256],
        "L_self_modules_branches_modules_2_modules_0_modules_linear1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_bias_",
    ),
    (
        [256, 1024],
        "L_self_modules_branches_modules_2_modules_0_modules_linear2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_2_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_2_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_branches_modules_2_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_branches_modules_2_modules_0_modules_self_attn_parameters_in_proj_weight_",
    ),
    ([1], "L_self_modules_branches_modules_2_modules_1_parameters_alpha_"),
    ([256], "L_self_modules_branches_modules_2_modules_2_modules_0_parameters_bias_"),
    (
        [256, 256],
        "L_self_modules_branches_modules_2_modules_2_modules_0_parameters_weight_",
    ),
    ([1], "L_self_modules_branches_modules_2_modules_2_modules_1_parameters_weight_"),
    ([1], "L_self_modules_branches_modules_2_modules_2_modules_2_parameters_bias_"),
    (
        [1, 256],
        "L_self_modules_branches_modules_2_modules_2_modules_2_parameters_weight_",
    ),
    ([1, 2499, 256], "L_stack0_"),
    ([], "s0"),
]
