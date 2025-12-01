from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([64], "L_self_modules_bn1_buffers_running_mean_"),
    ([64], "L_self_modules_bn1_buffers_running_var_"),
    ([64], "L_self_modules_bn1_parameters_bias_"),
    ([64], "L_self_modules_bn1_parameters_weight_"),
    ([64, 3, 7, 7], "L_self_modules_conv1_parameters_weight_"),
    ([1000], "L_self_modules_fc_parameters_bias_"),
    ([1000, 2048], "L_self_modules_fc_parameters_weight_"),
    ([112], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_"),
    ([112], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_"),
    ([112], "L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_"),
    ([112], "L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_weight_"),
    (
        [112, 64, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [256, 112, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_6_parameters_weight_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    ([112], "L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_"),
    ([112], "L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_"),
    ([112], "L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_"),
    ([112], "L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_weight_"),
    (
        [112, 256, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [256, 112, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_6_parameters_weight_",
    ),
    ([112], "L_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_"),
    ([112], "L_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_"),
    ([112], "L_self_modules_layer1_modules_2_modules_bn1_parameters_bias_"),
    ([112], "L_self_modules_layer1_modules_2_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_weight_"),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [14],
        "L_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_var_",
    ),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_bias_"),
    ([14], "L_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_weight_"),
    (
        [112, 256, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [256, 112, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [14, 14, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_6_parameters_weight_",
    ),
    ([224], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_"),
    ([224], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_"),
    ([224], "L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_weight_"),
    (
        [224, 256, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [512, 224, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_6_parameters_weight_",
    ),
    (
        [512, 256, 1, 1],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    ([224], "L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_"),
    ([224], "L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_"),
    ([224], "L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_weight_"),
    (
        [224, 512, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [512, 224, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_6_parameters_weight_",
    ),
    ([224], "L_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_"),
    ([224], "L_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_"),
    ([224], "L_self_modules_layer2_modules_2_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_layer2_modules_2_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_weight_"),
    (
        [224, 512, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [512, 224, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_6_parameters_weight_",
    ),
    ([224], "L_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_"),
    ([224], "L_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_"),
    ([224], "L_self_modules_layer2_modules_3_modules_bn1_parameters_bias_"),
    ([224], "L_self_modules_layer2_modules_3_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_weight_"),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [28],
        "L_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_var_",
    ),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_bias_"),
    ([28], "L_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_weight_"),
    (
        [224, 512, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [512, 224, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [28, 28, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_6_parameters_weight_",
    ),
    ([448], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_"),
    ([448], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_"),
    ([448], "L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_weight_"),
    (
        [448, 512, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 448, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_6_parameters_weight_",
    ),
    (
        [1024, 512, 1, 1],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    ([448], "L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_"),
    ([448], "L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_"),
    ([448], "L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_weight_"),
    (
        [448, 1024, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 448, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_6_parameters_weight_",
    ),
    ([448], "L_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_"),
    ([448], "L_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_"),
    ([448], "L_self_modules_layer3_modules_2_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_layer3_modules_2_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_weight_"),
    (
        [448, 1024, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 448, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_6_parameters_weight_",
    ),
    ([448], "L_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_"),
    ([448], "L_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_"),
    ([448], "L_self_modules_layer3_modules_3_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_layer3_modules_3_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_weight_"),
    (
        [448, 1024, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 448, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_6_parameters_weight_",
    ),
    ([448], "L_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_"),
    ([448], "L_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_"),
    ([448], "L_self_modules_layer3_modules_4_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_layer3_modules_4_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_weight_"),
    (
        [448, 1024, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 448, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_6_parameters_weight_",
    ),
    ([448], "L_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_"),
    ([448], "L_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_"),
    ([448], "L_self_modules_layer3_modules_5_modules_bn1_parameters_bias_"),
    ([448], "L_self_modules_layer3_modules_5_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_weight_"),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [56],
        "L_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_var_",
    ),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_bias_"),
    ([56], "L_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_weight_"),
    (
        [448, 1024, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 448, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [56, 56, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_6_parameters_weight_",
    ),
    ([896], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_"),
    ([896], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_"),
    ([896], "L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_weight_"),
    (
        [896, 1024, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [2048, 896, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_6_parameters_weight_",
    ),
    (
        [2048, 1024, 1, 1],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_",
    ),
    (
        [2048],
        "L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_",
    ),
    ([896], "L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_"),
    ([896], "L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_"),
    ([896], "L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_weight_"),
    (
        [896, 2048, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [2048, 896, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_6_parameters_weight_",
    ),
    ([896], "L_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_"),
    ([896], "L_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_"),
    ([896], "L_self_modules_layer4_modules_2_modules_bn1_parameters_bias_"),
    ([896], "L_self_modules_layer4_modules_2_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_weight_"),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_mean_",
    ),
    (
        [112],
        "L_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_var_",
    ),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_bias_"),
    ([112], "L_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_weight_"),
    (
        [896, 2048, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [2048, 896, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_3_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_4_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_5_parameters_weight_",
    ),
    (
        [112, 112, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_6_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
