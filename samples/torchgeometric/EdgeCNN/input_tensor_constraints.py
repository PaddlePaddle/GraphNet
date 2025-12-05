from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")
S2 = Symbol("S2")

dynamic_dim_constraint_symbols = [S0, S1, S2]

dynamic_dim_constraint_symbol2example_value = {S0: 2, S1: 100, S2: 1000}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1], "L_edge_index_"),
    (
        [256],
        "L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_convs_modules_0_modules_nn_modules_lins_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_bias_",
    ),
    (
        [256, 512],
        "L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_convs_modules_1_modules_nn_modules_lins_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_bias_",
    ),
    (
        [256, 512],
        "L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_convs_modules_2_modules_nn_modules_lins_modules_1_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_bias_",
    ),
    (
        [256, 512],
        "L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_convs_modules_3_modules_nn_modules_lins_modules_1_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_bias_",
    ),
    (
        [10, 512],
        "L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_0_parameters_weight_",
    ),
    (
        [10],
        "L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_bias_",
    ),
    (
        [10, 10],
        "L_self_modules_convs_modules_4_modules_nn_modules_lins_modules_1_parameters_weight_",
    ),
    ([S2, 128], "L_x_"),
]
