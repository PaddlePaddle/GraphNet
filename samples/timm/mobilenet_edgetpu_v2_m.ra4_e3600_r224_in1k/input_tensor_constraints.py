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
        [32, 64, 1, 1],
        "L_self_modules_blocks_modules_0_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_1_modules_0_modules_bn1_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_1_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_1_modules_0_modules_bn2_parameters_weight_"),
    (
        [256, 32, 3, 3],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_blocks_modules_1_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([256], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_bias_"),
    ([256], "L_self_modules_blocks_modules_1_modules_1_modules_bn1_parameters_weight_"),
    (
        [64],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_blocks_modules_1_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([64], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_bias_"),
    ([64], "L_self_modules_blocks_modules_1_modules_1_modules_bn2_parameters_weight_"),
    (
        [256, 16, 3, 3],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_exp_parameters_weight_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_blocks_modules_1_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_blocks_modules_2_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([512], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_bias_"),
    ([512], "L_self_modules_blocks_modules_2_modules_0_modules_bn1_parameters_weight_"),
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
        [512, 64, 3, 3],
        "L_self_modules_blocks_modules_2_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [80, 512, 1, 1],
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
        [320, 16, 3, 3],
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
        [320, 16, 3, 3],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_exp_parameters_weight_",
    ),
    (
        [80, 320, 1, 1],
        "L_self_modules_blocks_modules_2_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_0_modules_bn1_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_3_modules_0_modules_bn2_parameters_weight_"),
    (
        [640, 80, 3, 3],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_exp_parameters_weight_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_1_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_3_modules_1_modules_bn3_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_2_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_3_modules_2_modules_bn3_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn1_parameters_weight_"),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [640],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_bias_"),
    ([640], "L_self_modules_blocks_modules_3_modules_3_modules_bn2_parameters_weight_"),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [160],
        "L_self_modules_blocks_modules_3_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([160], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_bias_"),
    ([160], "L_self_modules_blocks_modules_3_modules_3_modules_bn3_parameters_weight_"),
    (
        [640, 1, 3, 3],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [640, 160, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [160, 640, 1, 1],
        "L_self_modules_blocks_modules_3_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1280], "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_bias_"),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1280], "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_bias_"),
    (
        [1280],
        "L_self_modules_blocks_modules_4_modules_0_modules_bn2_parameters_weight_",
    ),
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
        [1280, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1280, 160, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 1280, 1, 1],
        "L_self_modules_blocks_modules_4_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_1_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_2_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn1_parameters_weight_"),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [768],
        "L_self_modules_blocks_modules_4_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_bias_"),
    ([768], "L_self_modules_blocks_modules_4_modules_3_modules_bn2_parameters_weight_"),
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
        [768, 1, 3, 3],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [192, 768, 1, 1],
        "L_self_modules_blocks_modules_4_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1536], "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_bias_"),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1536], "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_bias_"),
    (
        [1536],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_0_modules_bn3_parameters_weight_"),
    (
        [1536, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1536, 192, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 1536, 1, 1],
        "L_self_modules_blocks_modules_5_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_1_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_1_modules_bn2_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_1_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_1_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_1_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_2_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_2_modules_bn2_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_2_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_2_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_2_modules_conv_pwl_parameters_weight_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn1_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_3_modules_bn1_parameters_weight_"),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_mean_",
    ),
    (
        [960],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn2_buffers_running_var_",
    ),
    ([960], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_bias_"),
    ([960], "L_self_modules_blocks_modules_5_modules_3_modules_bn2_parameters_weight_"),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_mean_",
    ),
    (
        [240],
        "L_self_modules_blocks_modules_5_modules_3_modules_bn3_buffers_running_var_",
    ),
    ([240], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_bias_"),
    ([240], "L_self_modules_blocks_modules_5_modules_3_modules_bn3_parameters_weight_"),
    (
        [960, 1, 3, 3],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_dw_parameters_weight_",
    ),
    (
        [960, 240, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pw_parameters_weight_",
    ),
    (
        [240, 960, 1, 1],
        "L_self_modules_blocks_modules_5_modules_3_modules_conv_pwl_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [1920],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn2_buffers_running_var_",
    ),
    ([1920], "L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_bias_"),
    (
        [1920],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_mean_",
    ),
    (
        [320],
        "L_self_modules_blocks_modules_6_modules_0_modules_bn3_buffers_running_var_",
    ),
    ([320], "L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_bias_"),
    ([320], "L_self_modules_blocks_modules_6_modules_0_modules_bn3_parameters_weight_"),
    (
        [1920, 1, 3, 3],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_dw_parameters_weight_",
    ),
    (
        [1920, 240, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_pw_parameters_weight_",
    ),
    (
        [320, 1920, 1, 1],
        "L_self_modules_blocks_modules_6_modules_0_modules_conv_pwl_parameters_weight_",
    ),
    ([64], "L_self_modules_bn1_buffers_running_mean_"),
    ([64], "L_self_modules_bn1_buffers_running_var_"),
    ([64], "L_self_modules_bn1_parameters_bias_"),
    ([64], "L_self_modules_bn1_parameters_weight_"),
    ([1344], "L_self_modules_bn2_buffers_running_mean_"),
    ([1344], "L_self_modules_bn2_buffers_running_var_"),
    ([1344], "L_self_modules_bn2_parameters_bias_"),
    ([1344], "L_self_modules_bn2_parameters_weight_"),
    ([1000], "L_self_modules_classifier_parameters_bias_"),
    ([1000, 1344], "L_self_modules_classifier_parameters_weight_"),
    ([1344, 320, 1, 1], "L_self_modules_conv_head_parameters_weight_"),
    ([64, 3, 5, 5], "L_self_modules_conv_stem_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
