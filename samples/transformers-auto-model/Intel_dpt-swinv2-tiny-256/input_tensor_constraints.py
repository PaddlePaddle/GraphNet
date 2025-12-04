from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([128], "L_self_modules_head_modules_head_modules_0_parameters_bias_"),
    ([128, 256, 3, 3], "L_self_modules_head_modules_head_modules_0_parameters_weight_"),
    ([], "L_self_modules_head_modules_head_modules_1_scale_factor"),
    ([32], "L_self_modules_head_modules_head_modules_2_parameters_bias_"),
    ([32, 128, 3, 3], "L_self_modules_head_modules_head_modules_2_parameters_weight_"),
    ([1], "L_self_modules_head_modules_head_modules_4_parameters_bias_"),
    ([1, 32, 1, 1], "L_self_modules_head_modules_head_modules_4_parameters_weight_"),
    ([256, 96, 3, 3], "L_self_modules_neck_modules_convs_modules_0_parameters_weight_"),
    (
        [256, 192, 3, 3],
        "L_self_modules_neck_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [256, 384, 3, 3],
        "L_self_modules_neck_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [256, 768, 3, 3],
        "L_self_modules_neck_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_projection_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_projection_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_projection_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_projection_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_bias_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_",
    ),
    ([S0, 96, 64, 64], "L_stack0_feature_maps_0_"),
    ([S0, 192, 32, 32], "L_stack0_feature_maps_1_"),
    ([S0, 384, 16, 16], "L_stack0_feature_maps_2_"),
    ([S0, 768, 8, 8], "L_stack0_feature_maps_3_"),
    ([], "s43"),
    ([], "s62"),
]
