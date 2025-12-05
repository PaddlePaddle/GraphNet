from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 512}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_inputs_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_2_"),
    (
        [1025, 1025],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_buffers_relative_position_index_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [3972, 12],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_",
    ),
    ([768], "L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_1_"),
    ([768], "L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_2_"),
    (
        [768],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_",
    ),
    (
        [768, 3, 16, 16],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_",
    ),
    ([1, 1, 768], "L_self_modules_backbone_parameters_cls_token_"),
    ([1, 1025, 768], "L_self_modules_backbone_parameters_pos_embed_"),
    (
        [768],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_",
    ),
    (
        [768, 3840, 3, 3],
        "L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_",
    ),
    ([150], "L_self_modules_decode_head_modules_conv_seg_parameters_bias_"),
    (
        [150, 768, 1, 1],
        "L_self_modules_decode_head_modules_conv_seg_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_",
    ),
    (
        [768, 3072, 3, 3],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [768, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_",
    ),
    ([768], "L_self_modules_neck_modules_upsample_2x_modules_0_parameters_bias_"),
    (
        [768, 768, 2, 2],
        "L_self_modules_neck_modules_upsample_2x_modules_0_parameters_weight_",
    ),
    ([768], "L_self_modules_neck_modules_upsample_4x_modules_0_parameters_bias_"),
    (
        [768, 768, 2, 2],
        "L_self_modules_neck_modules_upsample_4x_modules_0_parameters_weight_",
    ),
    ([768], "L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_"),
    ([768], "L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_"),
    ([768], "L_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_"),
    ([768], "L_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_"),
    ([768], "L_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_"),
    (
        [768, 768, 2, 2],
        "L_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_",
    ),
]
