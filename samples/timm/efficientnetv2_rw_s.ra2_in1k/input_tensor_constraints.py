from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [24, 24, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [24, 24, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_"),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [24],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_"),
    ([24], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_"),
    (
        [24, 24, 3, 3],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [24, 24, 1, 1],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([96], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([96], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [96, 24, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 96, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_"),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([48], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_"),
    ([48], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [48, 192, 1, 1],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [192, 48, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [64, 192, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [256, 64, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
    (
        [256, 64, 3, 3],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
    (
        [256, 64, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [256],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [256, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [256, 16, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [16, 256, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [512, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [512, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [512, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [512, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [512, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [512, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_"),
    (
        [512, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [512, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_"),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_"),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_"),
    (
        [512, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [512, 32, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 512, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 128, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [768, 32, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [32, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_"),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 160, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [960, 40, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [40],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [40, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_"),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_",
    ),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_"),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_",
    ),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_"),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_",
    ),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_"),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_",
    ),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_"),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_",
    ),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_",
    ),
    ([1632], "L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_"),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_",
    ),
    (
        [272],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_",
    ),
    ([272], "L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_"),
    ([272], "L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_"),
    (
        [1632, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_",
    ),
    (
        [1632, 272, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_",
    ),
    (
        [272, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1632],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1632, 68, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [68],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [68, 1632, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    ([24], "L_self_modules_bn1_buffers_running_mean_"),
    ([24], "L_self_modules_bn1_buffers_running_var_"),
    ([24], "L_self_modules_bn1_parameters_bias_"),
    ([24], "L_self_modules_bn1_parameters_weight_"),
    ([1792], "L_self_modules_bn2_buffers_running_mean_"),
    ([1792], "L_self_modules_bn2_buffers_running_var_"),
    ([1792], "L_self_modules_bn2_parameters_bias_"),
    ([1792], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1792], "L_self_modules_classifier_parameters_weight_"),
    ([1792, 272, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([24, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
