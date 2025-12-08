from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 3072], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([768], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([768], "L_self_modules_head_modules_norm_parameters_weight_"),
    ([3072], "L_self_modules_head_modules_pre_logits_modules_fc_parameters_bias_"),
    (
        [3072, 768],
        "L_self_modules_head_modules_pre_logits_modules_fc_parameters_weight_",
    ),
    ([3072], "L_self_modules_head_modules_pre_logits_modules_norm_parameters_bias_"),
    ([3072], "L_self_modules_head_modules_pre_logits_modules_norm_parameters_weight_"),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [682],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [682, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [128, 341],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [682],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [682, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [128, 341],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [682],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [682, 128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [128, 341],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_bias_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [1364],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [1364, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [256, 682],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_bias_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [1364],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [1364, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [256, 682],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_bias_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [1364],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [1364, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [256, 682],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_bias_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [1364],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_bias_",
    ),
    (
        [1364, 256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_bias_",
    ),
    (
        [256, 682],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_fc2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_stages_modules_1_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_stages_modules_1_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_10_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_11_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_12_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_13_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_14_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_15_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_16_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_17_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_18_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_19_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_20_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_21_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_22_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_23_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_24_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_25_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_26_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_6_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_7_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_8_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_conv_parameters_weight_",
    ),
    (
        [2730],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_bias_",
    ),
    (
        [2730, 512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_bias_",
    ),
    (
        [512, 1365],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_fc2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_blocks_modules_9_modules_norm_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_stages_modules_2_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_stages_modules_2_modules_downsample_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_bias_",
    ),
    (
        [768, 1, 7, 7],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_bias_",
    ),
    (
        [4096, 768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_bias_",
    ),
    (
        [768, 2048],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_fc2_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_bias_",
    ),
    (
        [768, 512, 3, 3],
        "L_self_modules_stages_modules_3_modules_downsample_modules_conv_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_stages_modules_3_modules_downsample_modules_norm_parameters_weight_",
    ),
    ([64], "L_self_modules_stem_modules_conv1_parameters_bias_"),
    ([64, 3, 3, 3], "L_self_modules_stem_modules_conv1_parameters_weight_"),
    ([128], "L_self_modules_stem_modules_conv2_parameters_bias_"),
    ([128, 64, 3, 3], "L_self_modules_stem_modules_conv2_parameters_weight_"),
    ([64], "L_self_modules_stem_modules_norm1_parameters_bias_"),
    ([64], "L_self_modules_stem_modules_norm1_parameters_weight_"),
    ([128], "L_self_modules_stem_modules_norm2_parameters_bias_"),
    ([128], "L_self_modules_stem_modules_norm2_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
