dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 31], "L_attention_mask_"),
    ([1, 31], "L_input_ids_"),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_0_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_10_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_11_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_1_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_2_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_3_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_4_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_5_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_6_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_7_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_8_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    (
        [4608],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_bias_",
    ),
    (
        [1536, 4608],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_attn_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_bias_",
    ),
    (
        [1536, 1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_attn_modules_c_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_ln_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_bias_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_ln_2_parameters_weight_",
    ),
    (
        [6144],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_bias_",
    ),
    (
        [1536, 6144],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_fc_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_bias_",
    ),
    (
        [6144, 1536],
        "L_self_modules_transformer_modules_h_modules_9_modules_mlp_modules_c_proj_parameters_weight_",
    ),
    ([1536], "L_self_modules_transformer_modules_ln_f_parameters_bias_"),
    ([1536], "L_self_modules_transformer_modules_ln_f_parameters_weight_"),
    ([2048, 1536], "L_self_modules_transformer_modules_wpe_parameters_weight_"),
    ([32002, 1536], "L_self_modules_transformer_modules_wte_parameters_weight_"),
]
