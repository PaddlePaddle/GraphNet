from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [144],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [144, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [144],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [144],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [576, 144, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 576, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_parameters_gamma_",
    ),
    (
        [272],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [272, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [272],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [272],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1088],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1088, 272, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1088, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_parameters_gamma_",
    ),
    (
        [400],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [400, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [400],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [400],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1600],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1600, 400, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1600, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_parameters_gamma_",
    ),
    ([1528], "L_self_modules_dense_stages_modules_10_modules_0_parameters_bias_"),
    ([1528], "L_self_modules_dense_stages_modules_10_modules_0_parameters_weight_"),
    ([760], "L_self_modules_dense_stages_modules_10_modules_1_parameters_bias_"),
    (
        [760, 1528, 2, 2],
        "L_self_modules_dense_stages_modules_10_modules_1_parameters_weight_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [760, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3040],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3040, 760, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [360, 3040, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [360, 360, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [1120],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1120, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1120],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1120],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4480],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4480, 1120, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [360, 4480, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [360, 360, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1480],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1480, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1480],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1480],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [5920],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [5920, 1480, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [360, 5920, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [360, 360, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1840], "L_self_modules_dense_stages_modules_11_modules_0_parameters_bias_"),
    ([1840], "L_self_modules_dense_stages_modules_11_modules_0_parameters_weight_"),
    ([920], "L_self_modules_dense_stages_modules_11_modules_1_parameters_bias_"),
    (
        [920, 1840, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_1_parameters_weight_",
    ),
    (
        [920],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [920, 1, 7, 7],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [920],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [920],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3680],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3680, 920, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [360, 3680, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [360, 360, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [1280],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1280, 1, 7, 7],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1280],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [5120],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [5120, 1280, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [360, 5120, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [360, 360, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1640],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1640, 1, 7, 7],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1640],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1640],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [6560],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [6560, 1640, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [360, 6560, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [360, 360, 1, 1],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_11_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([528], "L_self_modules_dense_stages_modules_1_modules_0_parameters_bias_"),
    ([528], "L_self_modules_dense_stages_modules_1_modules_0_parameters_weight_"),
    ([264], "L_self_modules_dense_stages_modules_1_modules_1_parameters_bias_"),
    (
        [264, 528, 2, 2],
        "L_self_modules_dense_stages_modules_1_modules_1_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [264, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [264],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [264],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1056],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1056, 264, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [192, 1056, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [456, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1824],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1824, 456, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [192, 1824, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [648],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [648, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [648],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [648],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2592],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2592, 648, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [192, 2592, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([840], "L_self_modules_dense_stages_modules_2_modules_0_parameters_bias_"),
    ([840], "L_self_modules_dense_stages_modules_2_modules_0_parameters_weight_"),
    ([416], "L_self_modules_dense_stages_modules_2_modules_1_parameters_bias_"),
    (
        [416, 840, 2, 2],
        "L_self_modules_dense_stages_modules_2_modules_1_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [416, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [416],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [416],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1664],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1664, 416, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 1664, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [672],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [672, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [672],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [672],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2688],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2688, 672, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 2688, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [928],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [928, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [928],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [928],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3712],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3712, 928, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 3712, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1184], "L_self_modules_dense_stages_modules_3_modules_0_parameters_bias_"),
    ([1184], "L_self_modules_dense_stages_modules_3_modules_0_parameters_weight_"),
    ([592], "L_self_modules_dense_stages_modules_3_modules_1_parameters_bias_"),
    (
        [592, 1184, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_1_parameters_weight_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [592, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2368],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2368, 592, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 2368, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [848],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [848, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [848],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [848],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3392],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3392, 848, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 3392, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1104],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1104, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1104],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1104],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4416],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4416, 1104, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 4416, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1360], "L_self_modules_dense_stages_modules_4_modules_0_parameters_bias_"),
    ([1360], "L_self_modules_dense_stages_modules_4_modules_0_parameters_weight_"),
    ([680], "L_self_modules_dense_stages_modules_4_modules_1_parameters_bias_"),
    (
        [680, 1360, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_1_parameters_weight_",
    ),
    (
        [680],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [680, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [680],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [680],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2720],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2720, 680, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 2720, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [936],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [936, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [936],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [936],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3744],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3744, 936, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 3744, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1192],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1192, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1192],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1192],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4768],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4768, 1192, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 4768, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1448], "L_self_modules_dense_stages_modules_5_modules_0_parameters_bias_"),
    ([1448], "L_self_modules_dense_stages_modules_5_modules_0_parameters_weight_"),
    ([720], "L_self_modules_dense_stages_modules_5_modules_1_parameters_bias_"),
    (
        [720, 1448, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_1_parameters_weight_",
    ),
    (
        [720],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [720, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [720],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [720],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2880],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2880, 720, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 2880, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [976],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [976, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [976],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [976],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3904],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3904, 976, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 3904, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1232],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1232, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1232],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1232],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4928],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4928, 1232, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 4928, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1488], "L_self_modules_dense_stages_modules_6_modules_0_parameters_bias_"),
    ([1488], "L_self_modules_dense_stages_modules_6_modules_0_parameters_weight_"),
    ([744], "L_self_modules_dense_stages_modules_6_modules_1_parameters_bias_"),
    (
        [744, 1488, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_1_parameters_weight_",
    ),
    (
        [744],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [744, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [744],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [744],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2976],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2976, 744, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 2976, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [1000],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1000, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1000],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1000],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4000],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4000, 1000, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 4000, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1256, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [5024],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [5024, 1256, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 5024, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1512], "L_self_modules_dense_stages_modules_7_modules_0_parameters_bias_"),
    ([1512], "L_self_modules_dense_stages_modules_7_modules_0_parameters_weight_"),
    ([752], "L_self_modules_dense_stages_modules_7_modules_1_parameters_bias_"),
    (
        [752, 1512, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_1_parameters_weight_",
    ),
    (
        [752],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [752, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [752],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [752],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3008],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3008, 752, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 3008, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [1008],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1008, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1008],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1008],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4032],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4032, 1008, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 4032, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1264],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1264, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1264],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1264],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [5056],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [5056, 1264, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 5056, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1520], "L_self_modules_dense_stages_modules_8_modules_0_parameters_bias_"),
    ([1520], "L_self_modules_dense_stages_modules_8_modules_0_parameters_weight_"),
    ([760], "L_self_modules_dense_stages_modules_8_modules_1_parameters_bias_"),
    (
        [760, 1520, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_1_parameters_weight_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [760, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3040],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3040, 760, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 3040, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [1016],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1016, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1016],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1016],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4064],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4064, 1016, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 4064, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1272],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1272, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1272],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1272],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [5088],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [5088, 1272, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 5088, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1528], "L_self_modules_dense_stages_modules_9_modules_0_parameters_bias_"),
    ([1528], "L_self_modules_dense_stages_modules_9_modules_0_parameters_weight_"),
    ([760], "L_self_modules_dense_stages_modules_9_modules_1_parameters_bias_"),
    (
        [760, 1528, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_1_parameters_weight_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [760, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [760],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3040],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3040, 760, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 3040, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [1016],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1016, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1016],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1016],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4064],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4064, 1016, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 4064, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1272],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1272, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1272],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1272],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [5088],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [5088, 1272, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [256, 5088, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [256, 256, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 2000], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([2000], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([2000], "L_self_modules_head_modules_norm_parameters_weight_"),
    ([144], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([144, 3, 4, 4], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([144], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([144], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
