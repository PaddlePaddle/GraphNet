from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 1}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [128],
        "L_self_modules_metric_head_modules_attractors_modules_0_modules_conv1_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [32],
        "L_self_modules_metric_head_modules_attractors_modules_0_modules_conv2_parameters_bias_",
    ),
    (
        [32, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_attractors_modules_1_modules_conv1_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_metric_head_modules_attractors_modules_1_modules_conv2_parameters_bias_",
    ),
    (
        [16, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_attractors_modules_2_modules_conv1_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [8],
        "L_self_modules_metric_head_modules_attractors_modules_2_modules_conv2_parameters_bias_",
    ),
    (
        [8, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_2_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_attractors_modules_3_modules_conv1_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [2],
        "L_self_modules_metric_head_modules_attractors_modules_3_modules_conv2_parameters_bias_",
    ),
    (
        [2, 128, 1, 1],
        "L_self_modules_metric_head_modules_attractors_modules_3_modules_conv2_parameters_weight_",
    ),
    (
        [1, 64, 1, 1],
        "L_self_modules_metric_head_modules_conditional_log_binomial_modules_log_binomial_transform_buffers_k_idx_",
    ),
    (
        [1, 1, 1, 1],
        "L_self_modules_metric_head_modules_conditional_log_binomial_modules_log_binomial_transform_buffers_k_minus_1_",
    ),
    (
        [80],
        "L_self_modules_metric_head_modules_conditional_log_binomial_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [80, 161, 1, 1],
        "L_self_modules_metric_head_modules_conditional_log_binomial_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_metric_head_modules_conditional_log_binomial_modules_mlp_modules_2_parameters_bias_",
    ),
    (
        [4, 80, 1, 1],
        "L_self_modules_metric_head_modules_conditional_log_binomial_modules_mlp_modules_2_parameters_weight_",
    ),
    ([256], "L_self_modules_metric_head_modules_conv2_parameters_bias_"),
    ([256, 256, 1, 1], "L_self_modules_metric_head_modules_conv2_parameters_weight_"),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_0_modules_conv1_parameters_bias_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_0_modules_conv2_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_0_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_1_modules_conv1_parameters_bias_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_1_modules_conv2_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_1_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_2_modules_conv1_parameters_bias_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_2_modules_conv2_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_2_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_3_modules_conv1_parameters_bias_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_projectors_modules_3_modules_conv2_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_projectors_modules_3_modules_conv2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_metric_head_modules_seed_bin_regressor_modules_conv1_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_metric_head_modules_seed_bin_regressor_modules_conv1_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_metric_head_modules_seed_bin_regressor_modules_conv2_parameters_bias_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_metric_head_modules_seed_bin_regressor_modules_conv2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_seed_projector_modules_conv1_parameters_bias_",
    ),
    (
        [128, 256, 1, 1],
        "L_self_modules_metric_head_modules_seed_projector_modules_conv1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_metric_head_modules_seed_projector_modules_conv2_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_metric_head_modules_seed_projector_modules_conv2_parameters_weight_",
    ),
    (
        [256, 256, 3, 3],
        "L_self_modules_neck_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [256, 512, 3, 3],
        "L_self_modules_neck_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [256, 1024, 3, 3],
        "L_self_modules_neck_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [256, 1024, 3, 3],
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
    (
        [256],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_bias_",
    ),
    (
        [256, 1024, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_projection_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_bias_",
    ),
    (
        [256, 256, 4, 4],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_0_modules_resize_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_bias_",
    ),
    (
        [512, 1024, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_projection_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_bias_",
    ),
    (
        [512, 512, 2, 2],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_1_modules_resize_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_2_modules_projection_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_bias_",
    ),
    (
        [1024, 1024, 1, 1],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_projection_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_bias_",
    ),
    (
        [1024, 1024, 3, 3],
        "L_self_modules_neck_modules_reassemble_stage_modules_layers_modules_3_modules_resize_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_bias_",
    ),
    (
        [1024, 2048],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_0_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_bias_",
    ),
    (
        [1024, 2048],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_1_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_bias_",
    ),
    (
        [1024, 2048],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_2_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_bias_",
    ),
    (
        [1024, 2048],
        "L_self_modules_neck_modules_reassemble_stage_modules_readout_projects_modules_3_modules_0_parameters_weight_",
    ),
    ([128], "L_self_modules_relative_head_modules_conv1_parameters_bias_"),
    ([128, 256, 3, 3], "L_self_modules_relative_head_modules_conv1_parameters_weight_"),
    ([32], "L_self_modules_relative_head_modules_conv2_parameters_bias_"),
    ([32, 128, 3, 3], "L_self_modules_relative_head_modules_conv2_parameters_weight_"),
    ([1], "L_self_modules_relative_head_modules_conv3_parameters_bias_"),
    ([1, 32, 1, 1], "L_self_modules_relative_head_modules_conv3_parameters_weight_"),
    ([S0, 1025, 1024], "L_stack0_feature_maps_0_"),
    ([S0, 1025, 1024], "L_stack0_feature_maps_1_"),
    ([S0, 1025, 1024], "L_stack0_feature_maps_2_"),
    ([S0, 1025, 1024], "L_stack0_feature_maps_3_"),
]
