from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 2048], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [1, 1, 3],
        "L_self_modules_stages_modules_0_modules_0_modules_attn_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [64, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_",
    ),
    (
        [64, 16, 3, 3],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_bn_parameters_weight_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_shortcut_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 3],
        "L_self_modules_stages_modules_0_modules_1_modules_attn_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_",
    ),
    (
        [64, 16, 3, 3],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_1_modules_0_modules_attn_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_",
    ),
    (
        [128, 16, 3, 3],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_bn_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_shortcut_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_1_modules_1_modules_attn_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_bn_parameters_weight_",
    ),
    (
        [128, 16, 3, 3],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_kxk_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_0_modules_attn_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [256, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_bn_parameters_weight_",
    ),
    (
        [256, 16, 3, 3],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_kxk_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_bn_parameters_weight_",
    ),
    (
        [1024, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_shortcut_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_post_attn_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_1_modules_post_attn_parameters_weight_",
    ),
    (
        [31, 16],
        "L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_",
    ),
    (
        [31, 16],
        "L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_",
    ),
    (
        [384, 256, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [512, 1024, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_post_attn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_0_modules_post_attn_parameters_weight_",
    ),
    (
        [31, 16],
        "L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_height_rel_",
    ),
    (
        [31, 16],
        "L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_pos_embed_parameters_width_rel_",
    ),
    (
        [640, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_mean_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_buffers_running_var_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_bn_parameters_weight_",
    ),
    (
        [2048, 1024, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_shortcut_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_bn_parameters_weight_",
    ),
    (
        [512, 2048, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_1x1_modules_conv_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_buffers_running_var_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_bn_parameters_weight_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_1x1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_post_attn_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_1_modules_post_attn_parameters_weight_",
    ),
    (
        [15, 16],
        "L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_height_rel_",
    ),
    (
        [15, 16],
        "L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_pos_embed_parameters_width_rel_",
    ),
    (
        [640, 512, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_self_attn_modules_qkv_parameters_weight_",
    ),
    ([24], "L_self_modules_stem_modules_conv1_modules_bn_buffers_running_mean_"),
    ([24], "L_self_modules_stem_modules_conv1_modules_bn_buffers_running_var_"),
    ([24], "L_self_modules_stem_modules_conv1_modules_bn_parameters_bias_"),
    ([24], "L_self_modules_stem_modules_conv1_modules_bn_parameters_weight_"),
    (
        [24, 3, 3, 3],
        "L_self_modules_stem_modules_conv1_modules_conv_parameters_weight_",
    ),
    ([32], "L_self_modules_stem_modules_conv2_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_stem_modules_conv2_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_stem_modules_conv2_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_stem_modules_conv2_modules_bn_parameters_weight_"),
    (
        [32, 24, 3, 3],
        "L_self_modules_stem_modules_conv2_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_stem_modules_conv3_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_stem_modules_conv3_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_stem_modules_conv3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_stem_modules_conv3_modules_bn_parameters_weight_"),
    (
        [64, 32, 3, 3],
        "L_self_modules_stem_modules_conv3_modules_conv_parameters_weight_",
    ),
    ([1, 3, 256, 256], "L_x_"),
]
