from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [72],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [72, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [72],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [72],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [288],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [288, 72, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [64, 288, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block0_parameters_gamma_",
    ),
    (
        [136],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [136, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [136],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [136],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [544],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [544, 136, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [64, 544, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block1_parameters_gamma_",
    ),
    (
        [200],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [200, 1, 7, 7],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [200],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [200],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [800],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [800, 200, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [64, 800, 1, 1],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [64],
        "L_self_modules_dense_stages_modules_0_modules_0_modules_dense_block2_parameters_gamma_",
    ),
    ([1096], "L_self_modules_dense_stages_modules_10_modules_0_parameters_bias_"),
    ([1096], "L_self_modules_dense_stages_modules_10_modules_0_parameters_weight_"),
    ([544], "L_self_modules_dense_stages_modules_10_modules_1_parameters_bias_"),
    (
        [544, 1096, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_1_parameters_weight_",
    ),
    (
        [544],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [544, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [544],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [544],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2176],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2176, 544, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [240, 2176, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [240, 240, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [784],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [784, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [784],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [784],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3136],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3136, 784, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [240, 3136, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [240, 240, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [1024],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [1024, 1, 7, 7],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4096],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4096, 1024, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [240, 4096, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [240, 240, 1, 1],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_10_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([264], "L_self_modules_dense_stages_modules_1_modules_0_parameters_bias_"),
    ([264], "L_self_modules_dense_stages_modules_1_modules_0_parameters_weight_"),
    ([128], "L_self_modules_dense_stages_modules_1_modules_1_parameters_bias_"),
    (
        [128, 264, 2, 2],
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
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 512, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1024, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [384],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [384, 1, 7, 7],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1536, 1, 1],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_1_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([512], "L_self_modules_dense_stages_modules_2_modules_0_parameters_bias_"),
    ([512], "L_self_modules_dense_stages_modules_2_modules_0_parameters_weight_"),
    ([256], "L_self_modules_dense_stages_modules_2_modules_1_parameters_bias_"),
    (
        [256, 512, 2, 2],
        "L_self_modules_dense_stages_modules_2_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1024, 256, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1024, 1, 1],
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
        [384],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [384, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [384],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [384],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1536],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1536, 384, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1536, 1, 1],
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
        [512],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [512, 1, 7, 7],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [512],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2048],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2048, 512, 1, 1],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_2_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2048, 1, 1],
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
    ([640], "L_self_modules_dense_stages_modules_3_modules_0_parameters_bias_"),
    ([640], "L_self_modules_dense_stages_modules_3_modules_0_parameters_weight_"),
    ([320], "L_self_modules_dense_stages_modules_3_modules_1_parameters_bias_"),
    (
        [320, 640, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_1_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [320, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [320],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [320],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1280],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1280, 320, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1280, 1, 1],
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
        [448],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [448, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [448],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [448],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1792],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1792, 448, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1792, 1, 1],
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
        [576],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [576, 1, 7, 7],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [576],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [576],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2304, 576, 1, 1],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_3_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2304, 1, 1],
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
    ([704], "L_self_modules_dense_stages_modules_4_modules_0_parameters_bias_"),
    ([704], "L_self_modules_dense_stages_modules_4_modules_0_parameters_weight_"),
    ([352], "L_self_modules_dense_stages_modules_4_modules_1_parameters_bias_"),
    (
        [352, 704, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_1_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [352, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [352],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [352],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1408],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1408, 352, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1408, 1, 1],
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
        [480],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [480, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [480],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [480],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1920],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1920, 480, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1920, 1, 1],
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
        [608],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [608, 1, 7, 7],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [608],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [608],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2432],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2432, 608, 1, 1],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_4_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2432, 1, 1],
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
    ([736], "L_self_modules_dense_stages_modules_5_modules_0_parameters_bias_"),
    ([736], "L_self_modules_dense_stages_modules_5_modules_0_parameters_weight_"),
    ([368], "L_self_modules_dense_stages_modules_5_modules_1_parameters_bias_"),
    (
        [368, 736, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_1_parameters_weight_",
    ),
    (
        [368],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [368, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [368],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [368],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1472],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1472, 368, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1472, 1, 1],
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
        [496],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [496, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [496],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1984],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1984, 496, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1984, 1, 1],
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
        [624],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [624, 1, 7, 7],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [624],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [624],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2496],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2496, 624, 1, 1],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_5_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2496, 1, 1],
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
    ([752], "L_self_modules_dense_stages_modules_6_modules_0_parameters_bias_"),
    ([752], "L_self_modules_dense_stages_modules_6_modules_0_parameters_weight_"),
    ([376], "L_self_modules_dense_stages_modules_6_modules_1_parameters_bias_"),
    (
        [376, 752, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_1_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [376, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1504],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1504, 376, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1504, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [504, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2016],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2016, 504, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2016, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [632, 1, 7, 7],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2528],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2528, 632, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2528, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_6_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([760], "L_self_modules_dense_stages_modules_7_modules_0_parameters_bias_"),
    ([760], "L_self_modules_dense_stages_modules_7_modules_0_parameters_weight_"),
    ([376], "L_self_modules_dense_stages_modules_7_modules_1_parameters_bias_"),
    (
        [376, 760, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_1_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [376, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1504],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1504, 376, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1504, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [504, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2016],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2016, 504, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2016, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [632, 1, 7, 7],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2528],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2528, 632, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2528, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_7_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([760], "L_self_modules_dense_stages_modules_8_modules_0_parameters_bias_"),
    ([760], "L_self_modules_dense_stages_modules_8_modules_0_parameters_weight_"),
    ([376], "L_self_modules_dense_stages_modules_8_modules_1_parameters_bias_"),
    (
        [376, 760, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_1_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [376, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1504],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1504, 376, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 1504, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [504, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [504],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2016],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2016, 504, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2016, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [632, 1, 7, 7],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [632],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2528],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2528, 632, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [128, 2528, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [128, 128, 1, 1],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [128],
        "L_self_modules_dense_stages_modules_8_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([760], "L_self_modules_dense_stages_modules_9_modules_0_parameters_bias_"),
    ([760], "L_self_modules_dense_stages_modules_9_modules_0_parameters_weight_"),
    ([376], "L_self_modules_dense_stages_modules_9_modules_1_parameters_bias_"),
    (
        [376, 760, 2, 2],
        "L_self_modules_dense_stages_modules_9_modules_1_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [376, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [376],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [1504],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [1504, 376, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [240, 1504, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [240, 240, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block0_parameters_gamma_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [616, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [616],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [2464],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [2464, 616, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [240, 2464, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [240, 240, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block1_parameters_gamma_",
    ),
    (
        [856],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [856, 1, 7, 7],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [856],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [856],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [3424],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [3424, 856, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_bias_",
    ),
    (
        [240, 3424, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_4_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_bias_",
    ),
    (
        [240, 240, 1, 1],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_modules_layers_modules_layers_modules_5_modules_fc_parameters_weight_",
    ),
    (
        [240],
        "L_self_modules_dense_stages_modules_9_modules_2_modules_dense_block2_parameters_gamma_",
    ),
    ([1000], "L_self_modules_head_modules_fc_parameters_bias_"),
    ([1000, 1264], "L_self_modules_head_modules_fc_parameters_weight_"),
    ([1264], "L_self_modules_head_modules_norm_parameters_bias_"),
    ([1264], "L_self_modules_head_modules_norm_parameters_weight_"),
    ([72], "L_self_modules_stem_modules_0_parameters_bias_"),
    ([72, 3, 4, 4], "L_self_modules_stem_modules_0_parameters_weight_"),
    ([72], "L_self_modules_stem_modules_1_parameters_bias_"),
    ([72], "L_self_modules_stem_modules_1_parameters_weight_"),
    ([S0, 3, S1, S1], "L_x_"),
    ([], "s1"),
]
