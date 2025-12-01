from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [64, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [256, 64, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [64, 256, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_parameters_gamma_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [64, 512, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_parameters_gamma_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [192, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [192],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [768, 192, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [64, 768, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_parameters_gamma_",
    ),
    ([256], "L_self_modules_dense_stages_modules_1_modules_0_parameters_bias_"),
    ([256], "L_self_modules_dense_stages_modules_1_modules_0_parameters_weight_"),
    ([128], "L_self_modules_dense_stages_modules_1_modules_1_parameters_bias_"),
    (
        [128, 256, 2, 2],
        "L_self_modules_dense_stages_modules_1_modules_1_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [128, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [512, 128, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [104, 512, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [232],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [232, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [232],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [232],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [928],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [928, 232, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [104, 928, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [336, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1344, 336, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [104, 1344, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [104],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([440], "L_self_modules_dense_stages_modules_2_modules_0_parameters_bias_"),
    ([440], "L_self_modules_dense_stages_modules_2_modules_0_parameters_weight_"),
    ([216], "L_self_modules_dense_stages_modules_2_modules_1_parameters_bias_"),
    (
        [216, 440, 2, 2],
        "L_self_modules_dense_stages_modules_2_modules_1_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [216, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [216],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [216],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [864],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [864, 216, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 864, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [344],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [344, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [344],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [344],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1376],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1376, 344, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1376, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [472],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [472, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [472],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [472],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1888],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1888, 472, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1888, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([600], "L_self_modules_dense_stages_modules_3_modules_0_parameters_bias_"),
    ([600], "L_self_modules_dense_stages_modules_3_modules_0_parameters_weight_"),
    ([296], "L_self_modules_dense_stages_modules_3_modules_1_parameters_bias_"),
    (
        [296, 600, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_1_parameters_weight_",
    ),
    (
        [296],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [296, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [296],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [296],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1184],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1184, 296, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1184, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [424],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [424, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [424],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [424],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1696],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1696, 424, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1696, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [552],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [552, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [552],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [552],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2208],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2208, 552, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2208, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([680], "L_self_modules_dense_stages_modules_4_modules_0_parameters_bias_"),
    ([680], "L_self_modules_dense_stages_modules_4_modules_0_parameters_weight_"),
    ([336], "L_self_modules_dense_stages_modules_4_modules_1_parameters_bias_"),
    (
        [336, 680, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_1_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [336, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [336],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1344],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1344, 336, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1344, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [464],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [464, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [464],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [464],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1856],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1856, 464, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1856, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [592, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2368],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2368, 592, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2368, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([720], "L_self_modules_dense_stages_modules_5_modules_0_parameters_bias_"),
    ([720], "L_self_modules_dense_stages_modules_5_modules_0_parameters_weight_"),
    ([360], "L_self_modules_dense_stages_modules_5_modules_1_parameters_bias_"),
    (
        [360, 720, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_1_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [360, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [360],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1440],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1440, 360, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1440, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [488],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [488, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [488],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [488],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1952],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1952, 488, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1952, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [616, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2464],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2464, 616, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2464, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([744], "L_self_modules_dense_stages_modules_6_modules_0_parameters_bias_"),
    ([744], "L_self_modules_dense_stages_modules_6_modules_0_parameters_weight_"),
    ([368], "L_self_modules_dense_stages_modules_6_modules_1_parameters_bias_"),
    (
        [368, 744, 2, 2],
        "L_self_modules_dense_stages_modules_6_modules_1_parameters_weight_",
    ),
    (
        [368],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [368, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [368],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [368],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1472],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1472, 368, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [224, 1472, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [224, 224, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [592, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [592],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2368],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2368, 592, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [224, 2368, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [224, 224, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [816],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [816, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [816],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [816],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3264],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3264, 816, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [224, 3264, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [224, 224, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [224],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 1040], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([1040], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([1040], "L_self_modules_head_modules_norm_parameters_weight_"),
    ([64], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([64, 3, 4, 4], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([64], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([64], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
