dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 19], "L_attention_mask_"),
    ([1, 19, 32], "L_inputs_embeds_"),
    ([512, 8], "L_self_modules_h_modules_0_modules_attn_embed_positions"),
    (
        [32, 32],
        "L_self_modules_h_modules_0_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_0_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_0_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_0_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_h_modules_0_modules_attn_scale_attn"),
    ([32], "L_self_modules_h_modules_0_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_0_modules_ln_1_parameters_weight_"),
    ([128], "L_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_0_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_0_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    ([512, 8], "L_self_modules_h_modules_1_modules_attn_embed_positions"),
    (
        [32, 32],
        "L_self_modules_h_modules_1_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_1_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_1_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_1_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_h_modules_1_modules_attn_scale_attn"),
    ([32], "L_self_modules_h_modules_1_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_1_modules_ln_1_parameters_weight_"),
    ([128], "L_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_1_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_1_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    ([512, 8], "L_self_modules_h_modules_2_modules_attn_embed_positions"),
    (
        [32, 32],
        "L_self_modules_h_modules_2_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_2_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_2_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_2_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_h_modules_2_modules_attn_scale_attn"),
    ([32], "L_self_modules_h_modules_2_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_2_modules_ln_1_parameters_weight_"),
    ([128], "L_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_2_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_2_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    ([512, 8], "L_self_modules_h_modules_3_modules_attn_embed_positions"),
    (
        [32, 32],
        "L_self_modules_h_modules_3_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_3_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_3_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_3_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_h_modules_3_modules_attn_scale_attn"),
    ([32], "L_self_modules_h_modules_3_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_3_modules_ln_1_parameters_weight_"),
    ([128], "L_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_3_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_3_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    ([512, 8], "L_self_modules_h_modules_4_modules_attn_embed_positions"),
    (
        [32, 32],
        "L_self_modules_h_modules_4_modules_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_4_modules_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_4_modules_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [32, 32],
        "L_self_modules_h_modules_4_modules_attn_modules_v_proj_parameters_weight_",
    ),
    ([], "L_self_modules_h_modules_4_modules_attn_scale_attn"),
    ([32], "L_self_modules_h_modules_4_modules_ln_1_parameters_bias_"),
    ([32], "L_self_modules_h_modules_4_modules_ln_1_parameters_weight_"),
    ([128], "L_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_bias_"),
    (
        [128, 32],
        "L_self_modules_h_modules_4_modules_mlp_modules_fc_in_parameters_weight_",
    ),
    ([32], "L_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_bias_"),
    (
        [32, 128],
        "L_self_modules_h_modules_4_modules_mlp_modules_fc_out_parameters_weight_",
    ),
    ([32], "L_self_modules_ln_f_parameters_bias_"),
    ([32], "L_self_modules_ln_f_parameters_weight_"),
]
