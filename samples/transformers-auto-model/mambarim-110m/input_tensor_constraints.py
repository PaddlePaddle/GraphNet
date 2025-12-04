from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 27], "L_attention_mask_"),
    ([S0, 27], "L_input_ids_"),
    ([32000, 768], "L_self_modules_backbone_modules_embeddings_parameters_weight_"),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_0_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_10_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_11_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_1_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_2_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_3_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_4_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_5_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_6_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_7_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_8_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_norm_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_modules_conv1d_parameters_bias_",
    ),
    (
        [1536, 1, 4],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_modules_conv1d_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_modules_dt_proj_parameters_bias_",
    ),
    (
        [1536, 48],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_modules_dt_proj_parameters_weight_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_modules_in_proj_parameters_weight_",
    ),
    (
        [768, 1536],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_modules_out_proj_parameters_weight_",
    ),
    (
        [80, 1536],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_modules_x_proj_parameters_weight_",
    ),
    (
        [1536, 16],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_parameters_A_log_",
    ),
    (
        [1536],
        "L_self_modules_backbone_modules_layers_modules_9_modules_mixer_parameters_D_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_norm_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_norm_f_parameters_weight_"),
]
