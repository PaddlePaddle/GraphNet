dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 384],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_blocks_modules_0_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_blocks_modules_0_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_blocks_modules_0_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 384],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_blocks_modules_1_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_blocks_modules_1_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_blocks_modules_1_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [384, 128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 384],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_0_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [768, 256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 768],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_blocks_modules_1_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wk_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wq_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_bias_",
    ),
    (
        [128, 128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_attn_modules_wv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_fusion_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_blocks_modules_2_modules_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_blocks_modules_2_modules_projs_modules_1_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_bias_",
    ),
    (
        [128, 256],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_0_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_bias_",
    ),
    (
        [256, 128],
        "L_self_modules_blocks_modules_2_modules_revert_projs_modules_1_modules_2_parameters_weight_",
    ),
    ([1000], "L_self_modules_head_modules_0_parameters_bias_"),
    ([1000, 128], "L_self_modules_head_modules_0_parameters_weight_"),
    ([1000], "L_self_modules_head_modules_1_parameters_bias_"),
    ([1000, 256], "L_self_modules_head_modules_1_parameters_weight_"),
    ([128], "L_self_modules_norm_modules_0_parameters_bias_"),
    ([128], "L_self_modules_norm_modules_0_parameters_weight_"),
    ([256], "L_self_modules_norm_modules_1_parameters_bias_"),
    ([256], "L_self_modules_norm_modules_1_parameters_weight_"),
    (
        [32],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_bias_",
    ),
    (
        [32, 3, 7, 7],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_bias_",
    ),
    (
        [64, 32, 3, 3],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_bias_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_patch_embed_modules_0_modules_proj_modules_4_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_bias_",
    ),
    (
        [64, 3, 7, 7],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_bias_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_bias_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_patch_embed_modules_1_modules_proj_modules_4_parameters_weight_",
    ),
    ([1, 1, 128], "L_self_parameters_cls_token_0_"),
    ([1, 1, 256], "L_self_parameters_cls_token_1_"),
    ([1, 401, 128], "L_self_parameters_pos_embed_0_"),
    ([1, 197, 256], "L_self_parameters_pos_embed_1_"),
    ([1, 3, 224, 224], "L_x_"),
]
