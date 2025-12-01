from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 512], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([512], "L_self_modules_norm_parameters_bias_"),
    ([512], "L_self_modules_norm_parameters_weight_"),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [64, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_ls_parameters_gamma_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_bias_",
    ),
    (
        [64, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_ls_parameters_gamma_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_bias_",
    ),
    (
        [64, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_ls_parameters_gamma_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_ls_parameters_gamma_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_",
    ),
    (
        [128, 64, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_ls_parameters_gamma_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_ls_parameters_gamma_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_dw_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_ls_parameters_gamma_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [384, 128, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [38416],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [729, 4],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [38416],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [729, 8],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [38416],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [729, 8],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [38416],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [729, 8],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [38416],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [729, 8],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [38416],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [729, 8],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [768, 256, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [2401],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [169, 8],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_bias_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_shortcut_modules_expand_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [2401],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [169, 16],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_bias_",
    ),
    (
        [512, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_proj_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_bias_",
    ),
    (
        [1536, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_qkv_parameters_weight_",
    ),
    (
        [2401],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_buffers_relative_position_index_",
    ),
    (
        [169, 16],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_attn_modules_rel_pos_parameters_relative_position_bias_table_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_bias_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_mlp_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_",
    ),
    ([32, 3, 3, 3], "L_self_modules_stem_modules_conv1_parameters_weight_"),
    ([64, 32, 3, 3], "L_self_modules_stem_modules_conv2_parameters_weight_"),
    ([32], "L_self_modules_stem_modules_norm1_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_norm1_parameters_weight_"),
    ([S0, 3, 224, 224], "L_x_"),
]
