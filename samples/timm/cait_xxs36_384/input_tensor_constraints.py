from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 384}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [4],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_0_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_0_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_0_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_0_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_0_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_0_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_0_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_10_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_10_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_10_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_10_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_10_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_10_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_10_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_10_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_10_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_11_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_11_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_11_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_11_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_11_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_11_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_11_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_11_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_11_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_12_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_12_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_12_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_12_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_12_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_12_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_12_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_12_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_12_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_12_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_12_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_12_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_13_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_13_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_13_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_13_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_13_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_13_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_13_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_13_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_13_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_13_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_13_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_13_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_14_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_14_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_14_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_14_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_14_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_14_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_14_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_14_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_14_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_14_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_14_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_14_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_15_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_15_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_15_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_15_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_15_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_15_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_15_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_15_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_15_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_15_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_15_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_15_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_16_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_16_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_16_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_16_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_16_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_16_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_16_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_16_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_16_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_16_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_16_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_16_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_17_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_17_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_17_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_17_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_17_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_17_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_17_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_17_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_17_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_17_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_17_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_17_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_18_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_18_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_18_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_18_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_18_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_18_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_18_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_18_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_18_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_18_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_18_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_18_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_19_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_19_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_19_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_19_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_19_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_19_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_19_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_19_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_19_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_19_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_19_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_19_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_1_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_1_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_1_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_1_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_20_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_20_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_20_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_20_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_20_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_20_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_20_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_20_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_20_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_20_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_20_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_20_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_21_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_21_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_21_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_21_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_21_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_21_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_21_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_21_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_21_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_21_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_21_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_21_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_22_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_22_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_22_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_22_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_22_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_22_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_22_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_22_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_22_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_22_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_22_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_22_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_23_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_23_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_23_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_23_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_23_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_23_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_23_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_23_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_23_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_23_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_23_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_23_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_24_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_24_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_24_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_24_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_24_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_24_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_24_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_24_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_24_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_24_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_24_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_24_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_24_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_24_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_25_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_25_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_25_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_25_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_25_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_25_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_25_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_25_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_25_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_25_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_25_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_25_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_25_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_25_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_26_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_26_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_26_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_26_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_26_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_26_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_26_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_26_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_26_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_26_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_26_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_26_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_26_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_26_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_27_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_27_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_27_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_27_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_27_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_27_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_27_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_27_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_27_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_27_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_27_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_27_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_27_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_27_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_28_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_28_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_28_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_28_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_28_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_28_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_28_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_28_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_28_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_28_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_28_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_28_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_28_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_28_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_29_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_29_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_29_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_29_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_29_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_29_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_29_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_29_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_29_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_29_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_29_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_29_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_29_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_29_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_2_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_2_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_2_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_2_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_30_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_30_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_30_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_30_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_30_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_30_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_30_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_30_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_30_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_30_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_30_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_30_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_30_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_30_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_31_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_31_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_31_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_31_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_31_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_31_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_31_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_31_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_31_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_31_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_31_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_31_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_31_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_31_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_32_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_32_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_32_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_32_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_32_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_32_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_32_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_32_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_32_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_32_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_32_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_32_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_32_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_32_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_33_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_33_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_33_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_33_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_33_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_33_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_33_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_33_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_33_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_33_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_33_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_33_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_33_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_33_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_34_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_34_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_34_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_34_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_34_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_34_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_34_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_34_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_34_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_34_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_34_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_34_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_34_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_34_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_35_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_35_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_35_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_35_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_35_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_35_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_modules_35_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_modules_35_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_35_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_35_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_35_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_35_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_35_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_35_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_3_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_3_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_3_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_3_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_3_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_3_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_3_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_4_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_4_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_4_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_4_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_5_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_5_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_5_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_5_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_5_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_5_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_5_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_6_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_6_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_6_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_6_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_6_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_6_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_6_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_6_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_6_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_7_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_7_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_7_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_7_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_7_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_7_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_7_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_7_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_7_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_8_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_8_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_8_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_8_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_8_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_8_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_8_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_8_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_8_parameters_gamma_2_"),
    (
        [4],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_l_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_bias_",
    ),
    (
        [4, 4],
        "L_self_modules_blocks_modules_9_modules_attn_modules_proj_w_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [576, 192],
        "L_self_modules_blocks_modules_9_modules_attn_modules_qkv_parameters_weight_",
    ),
    ([768], "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_bias_"),
    (
        [768, 192],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc1_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_bias_"),
    (
        [192, 768],
        "L_self_modules_blocks_modules_9_modules_mlp_modules_fc2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_modules_9_modules_norm1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_9_modules_norm1_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_9_modules_norm2_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_9_modules_norm2_parameters_weight_"),
    ([192], "L_self_modules_blocks_modules_9_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_modules_9_parameters_gamma_2_"),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_0_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_token_only_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_0_modules_norm2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_token_only_modules_0_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_token_only_modules_0_parameters_gamma_2_"),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_k_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_q_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_bias_",
    ),
    (
        [192, 192],
        "L_self_modules_blocks_token_only_modules_1_modules_attn_modules_v_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 192],
        "L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [192, 768],
        "L_self_modules_blocks_token_only_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_blocks_token_only_modules_1_modules_norm2_parameters_weight_",
    ),
    ([192], "L_self_modules_blocks_token_only_modules_1_parameters_gamma_1_"),
    ([192], "L_self_modules_blocks_token_only_modules_1_parameters_gamma_2_"),
    ([1000], "L_self_modules_head_parameters_bias_"),
    ([1000, 192], "L_self_modules_head_parameters_weight_"),
    ([192], "L_self_modules_norm_parameters_bias_"),
    ([192], "L_self_modules_norm_parameters_weight_"),
    ([192], "L_self_modules_patch_embed_modules_proj_parameters_bias_"),
    ([192, 3, 16, 16], "L_self_modules_patch_embed_modules_proj_parameters_weight_"),
    ([1, 1, 192], "L_self_parameters_cls_token_"),
    ([1, 576, 192], "L_self_parameters_pos_embed_"),
    ([1, 3, S0, S0], "L_x_"),
]
