from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 3, 512, 512], "L_inputs_"),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_backbone_modules_layer1_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_buffers_running_var_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [64, 64, 3, 3],
        "L_self_modules_backbone_modules_layer1_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [128, 64, 3, 3],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [128, 64, 1, 1],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [128, 128, 3, 3],
        "L_self_modules_backbone_modules_layer2_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [256, 128, 3, 3],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [256, 128, 1, 1],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_backbone_modules_layer3_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_bn2_parameters_weight_",
    ),
    (
        [512, 256, 3, 3],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_bn2_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_backbone_modules_layer4_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [32, 3, 3, 3],
        "L_self_modules_backbone_modules_stem_modules_0_parameters_weight_",
    ),
    ([32], "L_self_modules_backbone_modules_stem_modules_1_buffers_running_mean_"),
    ([32], "L_self_modules_backbone_modules_stem_modules_1_buffers_running_var_"),
    ([32], "L_self_modules_backbone_modules_stem_modules_1_parameters_bias_"),
    ([32], "L_self_modules_backbone_modules_stem_modules_1_parameters_weight_"),
    (
        [32, 32, 3, 3],
        "L_self_modules_backbone_modules_stem_modules_3_parameters_weight_",
    ),
    ([32], "L_self_modules_backbone_modules_stem_modules_4_buffers_running_mean_"),
    ([32], "L_self_modules_backbone_modules_stem_modules_4_buffers_running_var_"),
    ([32], "L_self_modules_backbone_modules_stem_modules_4_parameters_bias_"),
    ([32], "L_self_modules_backbone_modules_stem_modules_4_parameters_weight_"),
    (
        [64, 32, 3, 3],
        "L_self_modules_backbone_modules_stem_modules_6_parameters_weight_",
    ),
    ([64], "L_self_modules_backbone_modules_stem_modules_7_buffers_running_mean_"),
    ([64], "L_self_modules_backbone_modules_stem_modules_7_buffers_running_var_"),
    ([64], "L_self_modules_backbone_modules_stem_modules_7_parameters_bias_"),
    ([64], "L_self_modules_backbone_modules_stem_modules_7_parameters_weight_"),
    (
        [128],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_bottleneck_modules_bn_parameters_weight_",
    ),
    (
        [128, 1024, 3, 3],
        "L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_",
    ),
    ([6], "L_self_modules_decode_head_modules_conv_seg_parameters_bias_"),
    ([6, 128, 1, 1], "L_self_modules_decode_head_modules_conv_seg_parameters_weight_"),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_0_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_1_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_2_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_mean_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_buffers_running_var_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_bn_parameters_weight_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_modules_3_modules_1_modules_conv_parameters_weight_",
    ),
]
