from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")
S2 = Symbol("S2")

dynamic_dim_constraint_symbols = [S0, S1, S2]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 256, S2: 192}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, S1, S2], "L_inputs_"),
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_0_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_10_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_11_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_12_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_13_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_14_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_15_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_16_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_17_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_18_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_19_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_1_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_20_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_21_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_22_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_23_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_2_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_3_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_4_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_5_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_6_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_7_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_8_modules_attn_modules_qkv_parameters_weight_",
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
    (
        [1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [1024, 1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [3072, 1024],
        "L_self_modules_backbone_modules_layers_modules_9_modules_attn_modules_qkv_parameters_weight_",
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
    ([1024], "L_self_modules_backbone_modules_ln1_parameters_bias_"),
    ([1024], "L_self_modules_backbone_modules_ln1_parameters_weight_"),
    (
        [1024],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_bias_",
    ),
    (
        [1024, 3, 16, 16],
        "L_self_modules_backbone_modules_patch_embed_modules_projection_parameters_weight_",
    ),
    ([1, 192, 1024], "L_self_modules_backbone_parameters_pos_embed_"),
    (
        [1024, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_"),
    (
        [256, 256, 4, 4],
        "L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_",
    ),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_"),
    ([256], "L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_"),
    ([17], "L_self_modules_head_modules_final_layer_parameters_bias_"),
    ([17, 256, 1, 1], "L_self_modules_head_modules_final_layer_parameters_weight_"),
]
