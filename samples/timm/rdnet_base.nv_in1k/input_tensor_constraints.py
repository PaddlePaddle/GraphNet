from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [120],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [120, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [120],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [120],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [480, 120, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [96, 480, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_parameters_gamma_",
    ),
    (
        [216],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [216, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [216],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [864, 216, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [96, 864, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_parameters_gamma_",
    ),
    (
        [312],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [312, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [312],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [312],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1248],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1248, 312, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [96, 1248, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [96],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_parameters_gamma_",
    ),
    ([1504], "L_self_modules_dense_stages_modules_10_modules_0_parameters_bias_"),
    ([1504], "L_self_modules_dense_stages_modules_10_modules_0_parameters_weight_"),
    ([752], "L_self_modules_dense_stages_modules_10_modules_1_parameters_bias_"),
    (
        [752, 1504, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_1_parameters_weight_",
    ),
    (
        [752],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [752, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [752],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [752],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3008],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3008, 752, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [336, 3008, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [1088],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1088, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1088],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1088],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4352],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4352, 1088, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [336, 4352, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1424],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1424, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1424],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1424],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [5696],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [5696, 1424, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [336, 5696, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([408], "L_self_modules_dense_stages_modules_1_modules_0_parameters_bias_"),
    ([408], "L_self_modules_dense_stages_modules_1_modules_0_parameters_weight_"),
    ([200], "L_self_modules_dense_stages_modules_1_modules_1_parameters_bias_"),
    (
        [200, 408, 2, 2],
        "L_self_modules_dense_stages_modules_1_modules_1_parameters_weight_",
    ),
    (
        [200],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [200, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [200],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [200],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [800],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [800, 200, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 800, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [328],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [328, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [328],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [328],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1312],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1312, 328, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1312, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [456, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1824],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1824, 456, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1824, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([584], "L_self_modules_dense_stages_modules_2_modules_0_parameters_bias_"),
    ([584], "L_self_modules_dense_stages_modules_2_modules_0_parameters_weight_"),
    ([288], "L_self_modules_dense_stages_modules_2_modules_1_parameters_bias_"),
    (
        [288, 584, 2, 2],
        "L_self_modules_dense_stages_modules_2_modules_1_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [288, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [288],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1152],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1152, 288, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1152, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [456, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [456],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1824],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1824, 456, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1824, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [624],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [624, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [624],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2496],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2496, 624, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2496, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([792], "L_self_modules_dense_stages_modules_3_modules_0_parameters_bias_"),
    ([792], "L_self_modules_dense_stages_modules_3_modules_0_parameters_weight_"),
    ([392], "L_self_modules_dense_stages_modules_3_modules_1_parameters_bias_"),
    (
        [392, 792, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_1_parameters_weight_",
    ),
    (
        [392],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [392, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [392],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [392],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1568],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1568, 392, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1568, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [560],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [560, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [560],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [560],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2240],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2240, 560, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2240, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [728],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [728, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [728],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [728],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2912],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2912, 728, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2912, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([896], "L_self_modules_dense_stages_modules_4_modules_0_parameters_bias_"),
    ([896], "L_self_modules_dense_stages_modules_4_modules_0_parameters_weight_"),
    ([448], "L_self_modules_dense_stages_modules_4_modules_1_parameters_bias_"),
    (
        [448, 896, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_1_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [448, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1792],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1792, 448, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1792, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [616, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2464],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2464, 616, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2464, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [784],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [784, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [784],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3136],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3136, 784, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 3136, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([952], "L_self_modules_dense_stages_modules_5_modules_0_parameters_bias_"),
    ([952], "L_self_modules_dense_stages_modules_5_modules_0_parameters_weight_"),
    ([472], "L_self_modules_dense_stages_modules_5_modules_1_parameters_bias_"),
    (
        [472, 952, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_1_parameters_weight_",
    ),
    (
        [472],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [472, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [472],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [472],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1888],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1888, 472, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1888, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [640],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [640, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [640],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [640],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2560],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2560, 640, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2560, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [808],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [808, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [808],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [808],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3232],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3232, 808, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 3232, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([976], "L_self_modules_dense_stages_modules_6_modules_0_parameters_bias_"),
    ([976], "L_self_modules_dense_stages_modules_6_modules_0_parameters_weight_"),
    ([488], "L_self_modules_dense_stages_modules_6_modules_1_parameters_bias_"),
    (
        [488, 976, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_1_parameters_weight_",
    ),
    (
        [488],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [488, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [488],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [488],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1952],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1952, 488, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1952, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [656],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [656, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [656],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [656],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2624],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2624, 656, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2624, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [824],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [824, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [824],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [824],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3296],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3296, 824, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 3296, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([992], "L_self_modules_dense_stages_modules_7_modules_0_parameters_bias_"),
    ([992], "L_self_modules_dense_stages_modules_7_modules_0_parameters_weight_"),
    ([496], "L_self_modules_dense_stages_modules_7_modules_1_parameters_bias_"),
    (
        [496, 992, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_1_parameters_weight_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [496, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1984],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1984, 496, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1984, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [664],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [664, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [664],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [664],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2656],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2656, 664, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2656, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [832, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3328],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3328, 832, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 3328, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1000], "L_self_modules_dense_stages_modules_8_modules_0_parameters_bias_"),
    ([1000], "L_self_modules_dense_stages_modules_8_modules_0_parameters_weight_"),
    ([496], "L_self_modules_dense_stages_modules_8_modules_1_parameters_bias_"),
    (
        [496, 1000, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_1_parameters_weight_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [496, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1984],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1984, 496, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 1984, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [664],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [664, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [664],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [664],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2656],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2656, 664, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 2656, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [832, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3328],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3328, 832, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [168, 3328, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [168, 168, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [168],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1000], "L_self_modules_dense_stages_modules_9_modules_0_parameters_bias_"),
    ([1000], "L_self_modules_dense_stages_modules_9_modules_0_parameters_weight_"),
    ([496], "L_self_modules_dense_stages_modules_9_modules_1_parameters_bias_"),
    (
        [496, 1000, 2, 2],
        "L_self_modules_dense_stages_modules_9_modules_1_parameters_weight_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [496, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1984],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1984, 496, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [336, 1984, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [832, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [832],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3328],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3328, 832, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [336, 3328, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1168],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1168, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1168],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1168],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4672],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4672, 1168, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [336, 4672, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [336, 336, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 1760], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([1760], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([1760], "L_self_modules_head_modules_norm_parameters_weight_"),
    ([120], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([120, 3, 4, 4], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([120], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([120], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
