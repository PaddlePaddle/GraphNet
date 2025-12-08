from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 34, 1], "L_inputs_"),
    (
        [1024],
        "L_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_expand_conv_modules_bn_buffers_running_var_",
    ),
    ([1024], "L_self_modules_backbone_modules_expand_conv_modules_bn_parameters_bias_"),
    (
        [1024],
        "L_self_modules_backbone_modules_expand_conv_modules_bn_parameters_weight_",
    ),
    (
        [1024, 34, 1],
        "L_self_modules_backbone_modules_expand_conv_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1],
        "L_self_modules_backbone_modules_tcn_blocks_modules_0_modules_conv2_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv1_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_bn_parameters_weight_",
    ),
    (
        [1024, 1024, 1],
        "L_self_modules_backbone_modules_tcn_blocks_modules_1_modules_conv2_modules_0_modules_conv_parameters_weight_",
    ),
    ([48], "L_self_modules_head_modules_conv_parameters_bias_"),
    ([48, 1024, 1], "L_self_modules_head_modules_conv_parameters_weight_"),
]
