from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 43], "L_attention_mask_"),
    ([1, 43], "L_input_ids_"),
    ([512], "L_self_buffers_position_ids_"),
    ([1, 1, 512, 512], "L_self_modules_h_modules_0_modules_attn_buffers_bias_"),
    ([96], "L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_"),
    (
        [32, 96],
        "L_self_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_"),
    (
        [32, 32],
        "L_self_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_0_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_0_modules_ln_1_parameters_weight_"),
    ([32], "L_self_modules_h_modules_0_modules_ln_2_parameters_bias_"),
    ([32], "L_self_modules_h_modules_0_modules_ln_2_parameters_weight_"),
    ([128], "L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    ([1, 1, 512, 512], "L_self_modules_h_modules_1_modules_attn_buffers_bias_"),
    ([96], "L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_"),
    (
        [32, 96],
        "L_self_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_"),
    (
        [32, 32],
        "L_self_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_1_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_1_modules_ln_1_parameters_weight_"),
    ([32], "L_self_modules_h_modules_1_modules_ln_2_parameters_bias_"),
    ([32], "L_self_modules_h_modules_1_modules_ln_2_parameters_weight_"),
    ([128], "L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    ([1, 1, 512, 512], "L_self_modules_h_modules_2_modules_attn_buffers_bias_"),
    ([96], "L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_"),
    (
        [32, 96],
        "L_self_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_"),
    (
        [32, 32],
        "L_self_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_2_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_2_modules_ln_1_parameters_weight_"),
    ([32], "L_self_modules_h_modules_2_modules_ln_2_parameters_bias_"),
    ([32], "L_self_modules_h_modules_2_modules_ln_2_parameters_weight_"),
    ([128], "L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    ([1, 1, 512, 512], "L_self_modules_h_modules_3_modules_attn_buffers_bias_"),
    ([96], "L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_"),
    (
        [32, 96],
        "L_self_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_"),
    (
        [32, 32],
        "L_self_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_3_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_3_modules_ln_1_parameters_weight_"),
    ([32], "L_self_modules_h_modules_3_modules_ln_2_parameters_bias_"),
    ([32], "L_self_modules_h_modules_3_modules_ln_2_parameters_weight_"),
    ([128], "L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    ([1, 1, 512, 512], "L_self_modules_h_modules_4_modules_attn_buffers_bias_"),
    ([96], "L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_"),
    (
        [32, 96],
        "L_self_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_"),
    (
        [32, 32],
        "L_self_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_4_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_4_modules_ln_1_parameters_weight_"),
    ([32], "L_self_modules_h_modules_4_modules_ln_2_parameters_bias_"),
    ([32], "L_self_modules_h_modules_4_modules_ln_2_parameters_weight_"),
    ([128], "L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    ([512, 32], "L_self_modules_positions_embed_parameters_weight_"),
    ([1407, 32], "L_self_modules_tokens_embed_parameters_weight_"),
]
