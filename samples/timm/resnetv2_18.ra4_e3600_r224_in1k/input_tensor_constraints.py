from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 512, 1, 1], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([512], "L_self_modules_norm_buffers_running_mean_"),
    ([512], "L_self_modules_norm_buffers_running_var_"),
    ([512], "L_self_modules_norm_parameters_bias_"),
    ([512], "L_self_modules_norm_parameters_weight_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [128, 64, 1, 1],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_var_",
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
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_var_",
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
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_var_",
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
        [512, 256, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_var_",
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
        [512, 512, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_var_",
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
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_",
    ),
    ([64, 3, 7, 7], "L_self_modules_stem_modules_conv_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
