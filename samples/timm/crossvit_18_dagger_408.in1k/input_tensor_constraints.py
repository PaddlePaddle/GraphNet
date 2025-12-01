from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [672, 224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [672, 224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [224, 672],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [448, 224],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [224, 448],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [224, 448],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [448, 224],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [672, 224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [672, 224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [224, 672],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [448, 224],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [224, 448],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [224, 448],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [448, 224],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [672, 224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [672, 224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [224, 672],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1344, 448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [448, 1344],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [448, 448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [224, 224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [448, 224],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [224, 448],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [224, 448],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [448, 224],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    ([1000], "L_self_modules_head_modules_0_parameters_bias_"),
    ([1000, 224], "L_self_modules_head_modules_0_parameters_weight_"),
    ([1000], "L_self_modules_head_modules_1_parameters_bias_"),
    ([1000, 448], "L_self_modules_head_modules_1_parameters_weight_"),
    ([224], "L_self_modules_norm_modules_0_parameters_bias_"),
    ([224], "L_self_modules_norm_modules_0_parameters_weight_"),
    ([448], "L_self_modules_norm_modules_1_parameters_bias_"),
    ([448], "L_self_modules_norm_modules_1_parameters_weight_"),
    (
        [56],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_bias_",
    ),
    (
        [56, 3, 7, 7],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_weight_",
    ),
    (
        [112],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_bias_",
    ),
    (
        [112, 56, 3, 3],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_bias_",
    ),
    (
        [224, 112, 3, 3],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_weight_",
    ),
    (
        [112],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_bias_",
    ),
    (
        [112, 3, 7, 7],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_bias_",
    ),
    (
        [224, 112, 3, 3],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_bias_",
    ),
    (
        [448, 224, 3, 3],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_weight_",
    ),
    ([1, 1, 224], "L_self_parameters_cls_token_0_"),
    ([1, 1, 448], "L_self_parameters_cls_token_1_"),
    ([1, 1157, 224], "L_self_parameters_pos_embed_0_"),
    ([1, 577, 448], "L_self_parameters_pos_embed_1_"),
    ([1, 3, S0, S0], "L_x_"),
    ([], "s1"),
]
