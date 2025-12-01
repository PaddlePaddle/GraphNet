from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 640}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 3, S0, S0], "L_inputs_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_0_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_10_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_11_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_12_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_12_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_13_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_13_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_14_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_14_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_15_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_15_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_16_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_16_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_17_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_17_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_18_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_18_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_19_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_19_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_1_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_20_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_20_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_21_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_21_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_22_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_22_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_23_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_23_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_2_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_3_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_4_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_5_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_6_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_7_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_8_parameters_gamma_2_"),
    (
        [1601, 1601],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_buffers_relative_position_index_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_q_bias_",
    ),
    (
        [6244, 16],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_parameters_v_bias_",
    ),
    (
        [4096],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ffn_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_ln2_parameters_weight_",
    ),
    ([1024], "L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_1_"),
    ([1024], "L_self_modules_backbone_modules_layers_modules_9_parameters_gamma_2_"),
    (
        [1024],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_",
    ),
    (
        [1024, 3, 16, 16],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_",
    ),
    ([1, 1, 1024], "L_self_modules_backbone_parameters_cls_token_"),
    (
        [1024],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_",
    ),
    (
        [1024, 5120, 3, 3],
        "L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_",
    ),
    ([150], "L_self_modules_decode_head_modules_conv_seg_parameters_bias_"),
    (
        [150, 1024, 1, 1],
        "L_self_modules_decode_head_modules_conv_seg_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_bn_parameters_weight_",
    ),
    (
        [1024, 4096, 3, 3],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_",
    ),
    ([1024], "L_self_modules_neck_modules_upsample_2x_modules_0_parameters_bias_"),
    (
        [1024, 1024, 2, 2],
        "L_self_modules_neck_modules_upsample_2x_modules_0_parameters_weight_",
    ),
    ([1024], "L_self_modules_neck_modules_upsample_4x_modules_0_parameters_bias_"),
    (
        [1024, 1024, 2, 2],
        "L_self_modules_neck_modules_upsample_4x_modules_0_parameters_weight_",
    ),
    ([1024], "L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_mean_"),
    ([1024], "L_self_modules_neck_modules_upsample_4x_modules_1_buffers_running_var_"),
    ([1024], "L_self_modules_neck_modules_upsample_4x_modules_1_parameters_bias_"),
    ([1024], "L_self_modules_neck_modules_upsample_4x_modules_1_parameters_weight_"),
    ([1024], "L_self_modules_neck_modules_upsample_4x_modules_3_parameters_bias_"),
    (
        [1024, 1024, 2, 2],
        "L_self_modules_neck_modules_upsample_4x_modules_3_parameters_weight_",
    ),
]
