from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 256}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([3072], "L_self_modules_final_conv_parameters_bias_"),
    ([3072, 1, 1, 1], "L_self_modules_final_conv_parameters_gain_"),
    ([3072, 1536, 1, 1], "L_self_modules_final_conv_parameters_weight_"),
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 3072], "L_self_modules_head_modules_fc_parameters_weight_"),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_0_modules_0_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [64, 128, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv1_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_gain_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_0_modules_conv2b_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_stages_modules_0_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_0_modules_1_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv1_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2_parameters_weight_",
    ),
    ([64], "L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_bias_"),
    (
        [64, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_gain_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_stages_modules_0_modules_1_modules_conv2b_parameters_weight_",
    ),
    ([256], "L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_bias_"),
    (
        [256, 1, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_stages_modules_0_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_1_modules_0_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_bias_",
    ),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_0_modules_conv2b_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_stages_modules_1_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_1_modules_1_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_bias_",
    ),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_1_modules_conv2b_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_1_modules_2_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_gain_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_2_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_bias_",
    ),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_2_modules_conv2b_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_1_modules_3_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_gain_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv1_parameters_weight_",
    ),
    ([128], "L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_bias_"),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_3_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_bias_",
    ),
    (
        [128, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_gain_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_stages_modules_1_modules_3_modules_conv2b_parameters_weight_",
    ),
    ([512], "L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_bias_"),
    (
        [512, 1, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_gain_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_stages_modules_1_modules_3_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_0_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [384, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_0_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [1536, 512, 1, 1],
        "L_self_modules_stages_modules_2_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_10_modules_attn_last_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_10_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_10_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_10_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_10_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_11_modules_attn_last_modules_conv_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_11_modules_conv1_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_11_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_11_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_11_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_1_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_1_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_2_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_2_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_2_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_3_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_3_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_3_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_3_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_4_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_4_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_4_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_4_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_5_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_5_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_5_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_5_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_6_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_6_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_6_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_6_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_7_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_7_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_7_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_7_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_8_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_8_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_8_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_8_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_2_modules_9_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_2_modules_9_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_9_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_2_modules_9_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_2_modules_9_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_3_modules_0_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_0_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_gain_",
    ),
    (
        [1536, 1536, 1, 1],
        "L_self_modules_stages_modules_3_modules_0_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_3_modules_1_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_1_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_3_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_3_modules_2_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_2_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_2_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_3_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_3_modules_3_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_3_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_3_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_3_modules_3_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_3_modules_4_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_3_modules_4_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_4_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_4_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_3_modules_4_modules_conv3_parameters_weight_",
    ),
    (
        [1, 1, 5],
        "L_self_modules_stages_modules_3_modules_5_modules_attn_last_modules_conv_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_gain_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_stages_modules_3_modules_5_modules_conv1_parameters_weight_",
    ),
    ([384], "L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_bias_"),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_5_modules_conv2_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_bias_",
    ),
    (
        [384, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_gain_",
    ),
    (
        [384, 64, 3, 3],
        "L_self_modules_stages_modules_3_modules_5_modules_conv2b_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_bias_",
    ),
    (
        [1536, 1, 1, 1],
        "L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_gain_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_stages_modules_3_modules_5_modules_conv3_parameters_weight_",
    ),
    ([16], "L_self_modules_stem_modules_conv1_parameters_bias_"),
    ([16, 1, 1, 1], "L_self_modules_stem_modules_conv1_parameters_gain_"),
    ([16, 3, 3, 3], "L_self_modules_stem_modules_conv1_parameters_weight_"),
    ([32], "L_self_modules_stem_modules_conv2_parameters_bias_"),
    ([32, 1, 1, 1], "L_self_modules_stem_modules_conv2_parameters_gain_"),
    ([32, 16, 3, 3], "L_self_modules_stem_modules_conv2_parameters_weight_"),
    ([64], "L_self_modules_stem_modules_conv3_parameters_bias_"),
    ([64, 1, 1, 1], "L_self_modules_stem_modules_conv3_parameters_gain_"),
    ([64, 32, 3, 3], "L_self_modules_stem_modules_conv3_parameters_weight_"),
    ([128], "L_self_modules_stem_modules_conv4_parameters_bias_"),
    ([128, 1, 1, 1], "L_self_modules_stem_modules_conv4_parameters_gain_"),
    ([128, 64, 3, 3], "L_self_modules_stem_modules_conv4_parameters_weight_"),
    ([1, 3, S0, S0], "L_x_"),
]
