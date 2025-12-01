from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([64], "L_self_modules_conv_in_parameters_bias_"),
    ([64, 3, 3, 3], "L_self_modules_conv_in_parameters_weight_"),
    ([1], "L_self_modules_side1_parameters_bias_"),
    ([1, 64, 3, 3], "L_self_modules_side1_parameters_weight_"),
    ([1], "L_self_modules_side2_parameters_bias_"),
    ([1, 64, 3, 3], "L_self_modules_side2_parameters_weight_"),
    ([1], "L_self_modules_side3_parameters_bias_"),
    ([1, 128, 3, 3], "L_self_modules_side3_parameters_weight_"),
    ([1], "L_self_modules_side4_parameters_bias_"),
    ([1, 256, 3, 3], "L_self_modules_side4_parameters_weight_"),
    ([1], "L_self_modules_side5_parameters_bias_"),
    ([1, 512, 3, 3], "L_self_modules_side5_parameters_weight_"),
    ([1], "L_self_modules_side6_parameters_bias_"),
    ([1, 512, 3, 3], "L_self_modules_side6_parameters_weight_"),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [16, 64, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [64, 32, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [16, 16, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [16, 32, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [16, 16, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [16, 32, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [16, 16, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_bias_"),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_bias_",
    ),
    (
        [16, 32, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_bias_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_weight_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_bias_"),
    (
        [16, 16, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_bias_"),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_bias_",
    ),
    (
        [16, 32, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_bias_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_weight_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_bias_"),
    (
        [16, 16, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_bias_"),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_bias_",
    ),
    (
        [16, 32, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [16],
        "L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_var_",
    ),
    ([16], "L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_bias_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_weight_"),
    ([16], "L_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_bias_"),
    (
        [16, 16, 3, 3],
        "L_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [32, 128, 3, 3],
        "L_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_bias_"),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_bias_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_bias_"),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_bias_",
    ),
    (
        [32, 64, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_var_",
    ),
    ([32], "L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_bias_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_weight_"),
    ([32], "L_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_bias_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [64, 256, 3, 3],
        "L_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [64, 256, 3, 3],
        "L_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_bias_"),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_bias_"),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_bias_"),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_bias_"),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_bias_",
    ),
    (
        [64, 128, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_var_",
    ),
    ([64], "L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_bias_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_weight_"),
    ([64], "L_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_bias_"),
    (
        [64, 64, 3, 3],
        "L_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 512, 3, 3],
        "L_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([128], "L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([128], "L_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [128, 512, 3, 3],
        "L_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([128], "L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([128], "L_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [128, 128, 3, 3],
        "L_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([128], "L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([128], "L_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [128, 128, 3, 3],
        "L_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([128], "L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([128], "L_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [128, 128, 3, 3],
        "L_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 256, 3, 3],
        "L_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([128], "L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_bias_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 1024, 3, 3],
        "L_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 1024, 3, 3],
        "L_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_bias_"),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_bias_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_bias_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_bias_"),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_bias_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_var_",
    ),
    ([256], "L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_bias_"),
    ([256], "L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_weight_"),
    ([256], "L_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_bias_"),
    (
        [256, 256, 3, 3],
        "L_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_var_",
    ),
    ([512], "L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_bias_"),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_bias_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_weight_",
    ),
    ([S0, 3, 640, 640], "L_x_"),
]
