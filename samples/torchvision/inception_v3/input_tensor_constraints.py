from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")
S2 = Symbol("S2")

dynamic_dim_constraint_symbols = [S0, S1, S2]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 3, S2: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([32], "L_self_modules_Conv2d_1a_3x3_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_Conv2d_1a_3x3_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_Conv2d_1a_3x3_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_Conv2d_1a_3x3_modules_bn_parameters_weight_"),
    ([32, 3, 3, 3], "L_self_modules_Conv2d_1a_3x3_modules_conv_parameters_weight_"),
    ([32], "L_self_modules_Conv2d_2a_3x3_modules_bn_buffers_running_mean_"),
    ([32], "L_self_modules_Conv2d_2a_3x3_modules_bn_buffers_running_var_"),
    ([32], "L_self_modules_Conv2d_2a_3x3_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_Conv2d_2a_3x3_modules_bn_parameters_weight_"),
    ([32, 32, 3, 3], "L_self_modules_Conv2d_2a_3x3_modules_conv_parameters_weight_"),
    ([64], "L_self_modules_Conv2d_2b_3x3_modules_bn_buffers_running_mean_"),
    ([64], "L_self_modules_Conv2d_2b_3x3_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_Conv2d_2b_3x3_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Conv2d_2b_3x3_modules_bn_parameters_weight_"),
    ([64, 32, 3, 3], "L_self_modules_Conv2d_2b_3x3_modules_conv_parameters_weight_"),
    ([80], "L_self_modules_Conv2d_3b_1x1_modules_bn_buffers_running_mean_"),
    ([80], "L_self_modules_Conv2d_3b_1x1_modules_bn_buffers_running_var_"),
    ([80], "L_self_modules_Conv2d_3b_1x1_modules_bn_parameters_bias_"),
    ([80], "L_self_modules_Conv2d_3b_1x1_modules_bn_parameters_weight_"),
    ([80, 64, 1, 1], "L_self_modules_Conv2d_3b_1x1_modules_conv_parameters_weight_"),
    ([192], "L_self_modules_Conv2d_4a_3x3_modules_bn_buffers_running_mean_"),
    ([192], "L_self_modules_Conv2d_4a_3x3_modules_bn_buffers_running_var_"),
    ([192], "L_self_modules_Conv2d_4a_3x3_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_Conv2d_4a_3x3_modules_bn_parameters_weight_"),
    ([192, 80, 3, 3], "L_self_modules_Conv2d_4a_3x3_modules_conv_parameters_weight_"),
    (
        [64],
        "L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    ([64], "L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [64, 192, 1, 1],
        "L_self_modules_Mixed_5b_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 192, 1, 1],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [96, 64, 3, 3],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_parameters_weight_"),
    (
        [48, 192, 1, 1],
        "L_self_modules_Mixed_5b_modules_branch5x5_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_parameters_weight_"),
    (
        [64, 48, 5, 5],
        "L_self_modules_Mixed_5b_modules_branch5x5_2_modules_conv_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [32],
        "L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([32], "L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_parameters_bias_"),
    ([32], "L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_parameters_weight_"),
    (
        [32, 192, 1, 1],
        "L_self_modules_Mixed_5b_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    ([64], "L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [64, 256, 1, 1],
        "L_self_modules_Mixed_5c_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [96, 64, 3, 3],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_parameters_weight_"),
    (
        [48, 256, 1, 1],
        "L_self_modules_Mixed_5c_modules_branch5x5_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_parameters_weight_"),
    (
        [64, 48, 5, 5],
        "L_self_modules_Mixed_5c_modules_branch5x5_2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_parameters_weight_"),
    (
        [64, 256, 1, 1],
        "L_self_modules_Mixed_5c_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    ([64], "L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_buffers_running_var_"),
    ([64], "L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [64, 288, 1, 1],
        "L_self_modules_Mixed_5d_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 288, 1, 1],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [96, 64, 3, 3],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [48],
        "L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_mean_",
    ),
    (
        [48],
        "L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_var_",
    ),
    ([48], "L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_parameters_bias_"),
    ([48], "L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_parameters_weight_"),
    (
        [48, 288, 1, 1],
        "L_self_modules_Mixed_5d_modules_branch5x5_1_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_parameters_weight_"),
    (
        [64, 48, 5, 5],
        "L_self_modules_Mixed_5d_modules_branch5x5_2_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([64], "L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_parameters_bias_"),
    ([64], "L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_parameters_weight_"),
    (
        [64, 288, 1, 1],
        "L_self_modules_Mixed_5d_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_parameters_bias_"),
    ([384], "L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_parameters_weight_"),
    (
        [384, 288, 3, 3],
        "L_self_modules_Mixed_6a_modules_branch3x3_modules_conv_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [64, 288, 1, 1],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [96, 64, 3, 3],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [96],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [96, 96, 3, 3],
        "L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6b_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_Mixed_6b_modules_branch7x7_1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_var_",
    ),
    ([128], "L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_parameters_bias_"),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 7],
        "L_self_modules_Mixed_6b_modules_branch7x7_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_parameters_weight_",
    ),
    (
        [192, 128, 7, 1],
        "L_self_modules_Mixed_6b_modules_branch7x7_3_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 768, 1, 1],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 7, 1],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 1, 7],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_weight_",
    ),
    (
        [128, 128, 7, 1],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_weight_",
    ),
    (
        [192, 128, 1, 7],
        "L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6b_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6c_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_var_",
    ),
    ([160], "L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_parameters_bias_"),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_parameters_weight_",
    ),
    (
        [160, 768, 1, 1],
        "L_self_modules_Mixed_6c_modules_branch7x7_1_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_var_",
    ),
    ([160], "L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_parameters_bias_"),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 1, 7],
        "L_self_modules_Mixed_6c_modules_branch7x7_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_parameters_weight_",
    ),
    (
        [192, 160, 7, 1],
        "L_self_modules_Mixed_6c_modules_branch7x7_3_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [160, 768, 1, 1],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 7, 1],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 1, 7],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 7, 1],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_weight_",
    ),
    (
        [192, 160, 1, 7],
        "L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6c_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6d_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_var_",
    ),
    ([160], "L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_parameters_bias_"),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_parameters_weight_",
    ),
    (
        [160, 768, 1, 1],
        "L_self_modules_Mixed_6d_modules_branch7x7_1_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_var_",
    ),
    ([160], "L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_parameters_bias_"),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 1, 7],
        "L_self_modules_Mixed_6d_modules_branch7x7_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_parameters_weight_",
    ),
    (
        [192, 160, 7, 1],
        "L_self_modules_Mixed_6d_modules_branch7x7_3_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [160, 768, 1, 1],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 7, 1],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 1, 7],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_var_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_bias_",
    ),
    (
        [160],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_weight_",
    ),
    (
        [160, 160, 7, 1],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_weight_",
    ),
    (
        [192, 160, 1, 7],
        "L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6d_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_parameters_bias_"),
    ([192], "L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6e_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6e_modules_branch7x7_1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 1, 7],
        "L_self_modules_Mixed_6e_modules_branch7x7_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 7, 1],
        "L_self_modules_Mixed_6e_modules_branch7x7_3_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 7, 1],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 1, 7],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 7, 1],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 1, 7],
        "L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_6e_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_7a_modules_branch3x3_1_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_parameters_bias_"),
    (
        [320],
        "L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_parameters_weight_",
    ),
    (
        [320, 192, 3, 3],
        "L_self_modules_Mixed_7a_modules_branch3x3_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 1, 7],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 7, 1],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_var_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_weight_",
    ),
    (
        [192, 192, 3, 3],
        "L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [320, 1280, 1, 1],
        "L_self_modules_Mixed_7b_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_parameters_bias_"),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_parameters_weight_",
    ),
    (
        [384, 1280, 1, 1],
        "L_self_modules_Mixed_7b_modules_branch3x3_1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_parameters_bias_"),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 1, 3],
        "L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_parameters_bias_"),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 3, 1],
        "L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [448, 1280, 1, 1],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [384, 448, 3, 3],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 1, 3],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 3, 1],
        "L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_parameters_weight_",
    ),
    (
        [192, 1280, 1, 1],
        "L_self_modules_Mixed_7b_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_buffers_running_var_",
    ),
    ([320], "L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_parameters_bias_"),
    ([320], "L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_parameters_weight_"),
    (
        [320, 2048, 1, 1],
        "L_self_modules_Mixed_7c_modules_branch1x1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_parameters_bias_"),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_parameters_weight_",
    ),
    (
        [384, 2048, 1, 1],
        "L_self_modules_Mixed_7c_modules_branch3x3_1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_parameters_bias_"),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 1, 3],
        "L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_var_",
    ),
    ([384], "L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_parameters_bias_"),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 3, 1],
        "L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_conv_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_weight_",
    ),
    (
        [448, 2048, 1, 1],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_weight_",
    ),
    (
        [384, 448, 3, 3],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 1, 3],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_weight_",
    ),
    (
        [384, 384, 3, 1],
        "L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_conv_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_buffers_running_mean_",
    ),
    (
        [192],
        "L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_buffers_running_var_",
    ),
    ([192], "L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_parameters_bias_"),
    (
        [192],
        "L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_parameters_weight_",
    ),
    (
        [192, 2048, 1, 1],
        "L_self_modules_Mixed_7c_modules_branch_pool_modules_conv_parameters_weight_",
    ),
    ([1000], "L_self_modules_fc_parameters_bias_"),
    ([1000, 2048], "L_self_modules_fc_parameters_weight_"),
    ([S0, S1, S2, S2], "L_x_"),
    ([], "s0"),
    ([], "s1"),
]
