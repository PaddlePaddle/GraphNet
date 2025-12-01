from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([150], "L_self_modules_auxiliary_head_modules_classifier_parameters_bias_"),
    (
        [150, 256, 1, 1],
        "L_self_modules_auxiliary_head_modules_classifier_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_auxiliary_head_modules_convs_modules_0_modules_batch_norm_parameters_weight_",
    ),
    (
        [256, 384, 3, 3],
        "L_self_modules_auxiliary_head_modules_convs_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_bottleneck_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 2816, 3, 3],
        "L_self_modules_decode_head_modules_bottleneck_modules_conv_parameters_weight_",
    ),
    ([150], "L_self_modules_decode_head_modules_classifier_parameters_bias_"),
    (
        [150, 512, 1, 1],
        "L_self_modules_decode_head_modules_classifier_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 2048, 3, 3],
        "L_self_modules_decode_head_modules_fpn_bottleneck_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 512, 3, 3],
        "L_self_modules_decode_head_modules_fpn_convs_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 96, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_0_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 192, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 384, 1, 1],
        "L_self_modules_decode_head_modules_lateral_convs_modules_2_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_blocks_0_layers_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_blocks_1_layers_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_blocks_2_layers_1_modules_conv_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_batch_norm_parameters_weight_",
    ),
    (
        [512, 768, 1, 1],
        "L_self_modules_decode_head_modules_psp_modules_blocks_3_layers_1_modules_conv_parameters_weight_",
    ),
    ([S0, 96, 128, 128], "L_stack0_feature_maps_0_"),
    ([S0, 192, 64, 64], "L_stack0_feature_maps_1_"),
    ([S0, 384, 32, 32], "L_stack0_feature_maps_2_"),
    ([S0, 768, 16, 16], "L_stack0_feature_maps_3_"),
]
