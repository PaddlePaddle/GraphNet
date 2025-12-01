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
    ([104], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_"),
    ([104], "L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_"),
    ([104], "L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_"),
    ([104], "L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [104, 64, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [256, 104, 1, 1],
        "L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_0_modules_convs_modules_2_parameters_weight_",
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
    ([104], "L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_"),
    ([104], "L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_"),
    ([104], "L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_"),
    ([104], "L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [104, 256, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [256, 104, 1, 1],
        "L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    ([104], "L_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_"),
    ([104], "L_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_"),
    ([104], "L_self_modules_layer1_modules_2_modules_bn1_parameters_bias_"),
    ([104], "L_self_modules_layer1_modules_2_modules_bn1_parameters_weight_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_parameters_bias_"),
    ([256], "L_self_modules_layer1_modules_2_modules_bn3_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [26],
        "L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [26],
        "L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([26], "L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([26], "L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [104, 256, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [256, 104, 1, 1],
        "L_self_modules_layer1_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [26, 26, 3, 3],
        "L_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    ([208], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_"),
    ([208], "L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_"),
    ([208], "L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_"),
    ([208], "L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [208, 256, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [512, 208, 1, 1],
        "L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_",
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
    ([208], "L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_"),
    ([208], "L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_"),
    ([208], "L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_"),
    ([208], "L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [208, 512, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [512, 208, 1, 1],
        "L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    ([208], "L_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_"),
    ([208], "L_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_"),
    ([208], "L_self_modules_layer2_modules_2_modules_bn1_parameters_bias_"),
    ([208], "L_self_modules_layer2_modules_2_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_2_modules_bn3_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [208, 512, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [512, 208, 1, 1],
        "L_self_modules_layer2_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    ([208], "L_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_"),
    ([208], "L_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_"),
    ([208], "L_self_modules_layer2_modules_3_modules_bn1_parameters_bias_"),
    ([208], "L_self_modules_layer2_modules_3_modules_bn1_parameters_weight_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_parameters_bias_"),
    ([512], "L_self_modules_layer2_modules_3_modules_bn3_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_"),
    (
        [52],
        "L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [52],
        "L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_",
    ),
    ([52], "L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_"),
    ([52], "L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_"),
    (
        [208, 512, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [512, 208, 1, 1],
        "L_self_modules_layer2_modules_3_modules_conv3_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [52, 52, 3, 3],
        "L_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 512, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_",
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
    ([416], "L_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_10_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_10_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_10_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_10_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_10_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_10_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_10_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_10_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_10_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_11_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_11_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_11_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_11_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_11_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_11_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_11_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_11_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_11_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_12_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_12_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_12_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_12_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_12_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_12_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_12_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_12_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_12_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_13_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_13_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_13_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_13_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_13_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_13_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_13_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_13_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_13_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_14_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_14_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_14_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_14_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_14_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_14_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_14_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_14_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_14_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_15_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_15_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_15_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_15_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_15_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_15_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_15_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_15_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_15_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_16_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_16_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_16_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_16_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_16_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_16_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_16_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_16_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_16_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_17_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_17_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_17_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_17_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_17_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_17_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_17_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_17_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_17_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_18_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_18_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_18_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_18_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_18_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_18_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_18_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_18_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_18_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_19_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_19_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_19_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_19_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_19_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_19_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_19_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_19_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_19_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_20_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_20_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_20_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_20_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_20_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_20_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_20_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_20_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_20_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_21_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_21_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_21_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_21_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_21_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_21_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_21_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_21_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_21_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_22_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_22_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_22_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_22_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_bias_"),
    (
        [104],
        "L_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_weight_",
    ),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_22_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_22_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_22_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_22_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_22_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_2_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_2_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_2_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_3_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_3_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_3_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_3_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_4_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_4_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_4_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_4_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_5_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_5_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_6_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_6_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_6_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_6_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_6_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_6_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_6_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_6_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_6_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_7_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_7_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_7_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_7_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_7_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_7_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_7_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_7_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_7_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_8_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_8_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_8_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_8_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_8_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_8_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_8_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_8_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_8_modules_convs_modules_2_parameters_weight_",
    ),
    ([416], "L_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_"),
    ([416], "L_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_"),
    ([416], "L_self_modules_layer3_modules_9_modules_bn1_parameters_bias_"),
    ([416], "L_self_modules_layer3_modules_9_modules_bn1_parameters_weight_"),
    ([1024], "L_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_"),
    ([1024], "L_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_"),
    ([1024], "L_self_modules_layer3_modules_9_modules_bn3_parameters_bias_"),
    ([1024], "L_self_modules_layer3_modules_9_modules_bn3_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_weight_"),
    (
        [104],
        "L_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [104],
        "L_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_var_",
    ),
    ([104], "L_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_bias_"),
    ([104], "L_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_weight_"),
    (
        [416, 1024, 1, 1],
        "L_self_modules_layer3_modules_9_modules_conv1_parameters_weight_",
    ),
    (
        [1024, 416, 1, 1],
        "L_self_modules_layer3_modules_9_modules_conv3_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_9_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_9_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [104, 104, 3, 3],
        "L_self_modules_layer3_modules_9_modules_convs_modules_2_parameters_weight_",
    ),
    ([832], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_"),
    ([832], "L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_"),
    ([832], "L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_"),
    ([832], "L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_"),
    (
        [832, 1024, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_",
    ),
    (
        [2048, 832, 1, 1],
        "L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_",
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
    ([832], "L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_"),
    ([832], "L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_"),
    ([832], "L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_"),
    ([832], "L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_"),
    (
        [832, 2048, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_",
    ),
    (
        [2048, 832, 1, 1],
        "L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_",
    ),
    ([832], "L_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_"),
    ([832], "L_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_"),
    ([832], "L_self_modules_layer4_modules_2_modules_bn1_parameters_bias_"),
    ([832], "L_self_modules_layer4_modules_2_modules_bn1_parameters_weight_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_"),
    ([2048], "L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_"),
    (
        [208],
        "L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_",
    ),
    (
        [208],
        "L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_",
    ),
    ([208], "L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_"),
    ([208], "L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_"),
    (
        [832, 2048, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv1_parameters_weight_",
    ),
    (
        [2048, 832, 1, 1],
        "L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_",
    ),
    (
        [208, 208, 3, 3],
        "L_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_",
    ),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
