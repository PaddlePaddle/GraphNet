from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn1_parameters_weight_"),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_0_modules_bn2_parameters_weight_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [32, 32, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_1_modules_bn1_parameters_weight_"),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_1_modules_bn2_parameters_weight_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [32, 32, 1, 1],
        "L_self_modules_blocks_modules_0_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_2_modules_bn1_parameters_weight_"),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_blocks_modules_0_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([32], "L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_bias_"),
    ([32], "L_self_modules_blocks_modules_0_modules_2_modules_bn2_parameters_weight_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_blocks_modules_0_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [32, 32, 1, 1],
        "L_self_modules_blocks_modules_0_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([128], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([128], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [128, 32, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 128, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([224], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [224, 56, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 224, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([224], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_blocks_modules_1_modules_2_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_1_modules_2_modules_bn2_parameters_weight_"),
    (
        [224, 56, 3, 3],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 224, 1, 1],
        "L_self_modules_blocks_modules_1_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([224], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_blocks_modules_1_modules_3_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_1_modules_3_modules_bn2_parameters_weight_"),
    (
        [224, 56, 3, 3],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 224, 1, 1],
        "L_self_modules_blocks_modules_1_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([224], "L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_blocks_modules_1_modules_4_modules_bn1_parameters_weight_"),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_blocks_modules_1_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([56], "L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_bias_"),
    ([56], "L_self_modules_blocks_modules_1_modules_4_modules_bn2_parameters_weight_"),
    (
        [224, 56, 3, 3],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_exp_parameters_weight_",
    ),
    (
        [56, 224, 1, 1],
        "L_self_modules_blocks_modules_1_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [224],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([224], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_2_modules_0_modules_bn2_parameters_weight_"),
    (
        [224, 56, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [80, 224, 1, 1],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_1_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_2_modules_1_modules_bn2_parameters_weight_"),
    (
        [320, 80, 3, 3],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_blocks_modules_2_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_2_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_2_modules_2_modules_bn2_parameters_weight_"),
    (
        [320, 80, 3, 3],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_exp_parameters_weight_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_blocks_modules_2_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_3_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_2_modules_3_modules_bn2_parameters_weight_"),
    (
        [320, 80, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_2_modules_4_modules_bn1_parameters_weight_"),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [80],
        "L_self_modules_blocks_modules_2_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([80], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_bias_"),
    ([80], "L_self_modules_blocks_modules_2_modules_4_modules_bn2_parameters_weight_"),
    (
        [320, 80, 3, 3],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_exp_parameters_weight_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_blocks_modules_2_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [320],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_0_modules_bn3_parameters_weight_"),
    (
        [320, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [320, 80, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 320, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [320, 20, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [20],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [20, 320, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [608, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [608, 152, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [608, 38, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [608, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [608, 152, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [608, 38, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [608, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [608, 152, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [608, 38, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_4_modules_bn1_parameters_weight_"),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_4_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_4_modules_bn3_parameters_weight_"),
    (
        [608, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [608, 152, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [608, 38, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_5_modules_bn1_parameters_weight_"),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_5_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_5_modules_bn3_parameters_weight_"),
    (
        [608, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [608, 152, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [608, 38, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_6_modules_bn1_parameters_weight_"),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_6_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_6_modules_bn3_parameters_weight_"),
    (
        [608, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [608, 152, 1, 1],
        "L_self_modules_blocks_modules_3_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [608, 38, 1, 1],
        "L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_7_modules_bn1_parameters_weight_"),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([608], "L_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_bias_"),
    ([608], "L_self_modules_blocks_modules_3_modules_7_modules_bn2_parameters_weight_"),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [152],
        "L_self_modules_blocks_modules_3_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([152], "L_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_bias_"),
    ([152], "L_self_modules_blocks_modules_3_modules_7_modules_bn3_parameters_weight_"),
    (
        [608, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [608, 152, 1, 1],
        "L_self_modules_blocks_modules_3_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [152, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [608, 38, 1, 1],
        "L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 608, 1, 1],
        "L_self_modules_blocks_modules_3_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([912], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    ([912], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_"),
    (
        [912],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [912],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([912], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    ([912], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_0_modules_bn3_parameters_weight_"),
    (
        [912, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [912, 152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 912, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [912],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [912, 38, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [38],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [38, 912, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_bias_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_10_modules_bn3_parameters_weight_",
    ),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_10_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_10_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_10_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_10_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_bias_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_11_modules_bn3_parameters_weight_",
    ),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_11_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_11_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_11_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_11_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_bias_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_12_modules_bn3_parameters_weight_",
    ),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_12_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_12_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_12_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_12_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_bias_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_13_modules_bn3_parameters_weight_",
    ),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_13_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_13_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_13_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_13_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_bias_"),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_14_modules_bn3_parameters_weight_",
    ),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_14_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_14_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_14_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_14_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_1_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_2_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_3_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_4_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_5_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_6_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_7_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_8_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_blocks_modules_4_modules_9_modules_bn3_buffers_running_var_",
    ),
    ([192], "L_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_bias_"),
    ([192], "L_self_modules_blocks_modules_4_modules_9_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_9_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_9_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_9_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_4_modules_9_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1152], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [1152, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1152, 192, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1152, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1152, 48, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [48, 1152, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_10_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_10_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_11_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_11_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_12_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_12_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_13_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_13_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_14_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_14_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_14_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_15_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_15_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_15_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_15_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_15_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_16_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_16_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_16_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_16_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_16_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_17_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_17_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_17_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_17_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_17_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_18_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_18_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_18_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_18_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_18_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_19_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_19_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_19_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_19_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_19_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_20_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_20_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_20_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_20_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_20_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_21_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_21_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_21_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_21_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_21_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_22_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_22_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_22_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_22_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_22_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_bias_"),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_23_modules_bn3_parameters_weight_",
    ),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_23_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_23_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_23_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_23_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_4_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_4_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_4_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_5_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_5_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_5_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_6_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_6_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_6_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_7_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_7_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_7_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_8_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_8_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_8_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn1_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_mean_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_buffers_running_var_",
    ),
    ([1968], "L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_bias_"),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn2_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_mean_",
    ),
    (
        [328],
        "L_self_modules_blocks_modules_5_modules_9_modules_bn3_buffers_running_var_",
    ),
    ([328], "L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_bias_"),
    ([328], "L_self_modules_blocks_modules_5_modules_9_modules_bn3_parameters_weight_"),
    (
        [1968, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_dw_parameters_weight_",
    ),
    (
        [1968, 328, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_pw_parameters_weight_",
    ),
    (
        [328, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1968],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_bias_",
    ),
    (
        [1968, 82, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_expand_parameters_weight_",
    ),
    (
        [82],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_bias_",
    ),
    (
        [82, 1968, 1, 1],
        "L_self_modules_blocks_modules_5_modules_9_modules_se_modules_conv_reduce_parameters_weight_",
    ),
    ([32], "L_self_modules_bn1_buffers_running_mean_"),
    ([32], "L_self_modules_bn1_buffers_running_var_"),
    ([32], "L_self_modules_bn1_parameters_bias_"),
    ([32], "L_self_modules_bn1_parameters_weight_"),
    ([2152], "L_self_modules_bn2_buffers_running_mean_"),
    ([2152], "L_self_modules_bn2_buffers_running_var_"),
    ([2152], "L_self_modules_bn2_parameters_bias_"),
    ([2152], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 2152], "L_self_modules_classifier_parameters_weight_"),
    ([2152, 328, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
