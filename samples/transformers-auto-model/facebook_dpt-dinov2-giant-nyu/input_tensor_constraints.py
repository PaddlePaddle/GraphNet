from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([128], "L_self_modules_head_modules_head_modules_0_parameters_bias_"),
    ([128, 256, 3, 3], "L_self_modules_head_modules_head_modules_0_parameters_weight_"),
    ([32], "L_self_modules_head_modules_head_modules_2_parameters_bias_"),
    ([32, 128, 3, 3], "L_self_modules_head_modules_head_modules_2_parameters_weight_"),
    ([1], "L_self_modules_head_modules_head_modules_4_parameters_bias_"),
    ([1, 32, 1, 1], "L_self_modules_head_modules_head_modules_4_parameters_weight_"),
    ([256], "L_self_modules_head_modules_projection_parameters_bias_"),
    ([256, 256, 3, 3], "L_self_modules_head_modules_projection_parameters_weight_"),
    (
        [256, 192, 3, 3],
        "L_self_modules_neck_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [256, 384, 3, 3],
        "L_self_modules_neck_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [256, 768, 3, 3],
        "L_self_modules_neck_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [256, 1536, 3, 3],
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
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_0_modules_residual_layer2_modules_convolution1_parameters_weight_",
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
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer1_modules_convolution2_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_1_modules_residual_layer2_modules_convolution1_parameters_weight_",
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
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer1_modules_convolution2_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_2_modules_residual_layer2_modules_convolution1_parameters_weight_",
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
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer1_modules_convolution2_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution1_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_fusion_stage_modules_layers_modules_3_modules_residual_layer2_modules_convolution2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_",
    ),
    (
        [192, 1536, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_",
    ),
    (
        [192, 192, 4, 4],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_",
    ),
    (
        [384, 1536, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_",
    ),
    (
        [384, 384, 2, 2],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_",
    ),
    (
        [768, 1536, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_",
    ),
    (
        [1536, 1536, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_",
    ),
    (
        [1536, 1536, 3, 3],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_bias_",
    ),
    (
        [1536, 3072],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_bias_",
    ),
    (
        [1536, 3072],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_bias_",
    ),
    (
        [1536, 3072],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_bias_",
    ),
    (
        [1536, 3072],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_weight_",
    ),
    ([1, 785, 1536], "L_stack0_feature_maps_0_"),
    ([1, 785, 1536], "L_stack0_feature_maps_1_"),
    ([1, 785, 1536], "L_stack0_feature_maps_2_"),
    ([1, 785, 1536], "L_stack0_feature_maps_3_"),
]
