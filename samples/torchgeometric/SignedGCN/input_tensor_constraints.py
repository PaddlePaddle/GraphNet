from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 2}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 100], "L_neg_edge_index_"),
    ([S0, 100], "L_pos_edge_index_"),
    ([32, 32], "L_self_modules_conv1_modules_lin_neg_l_parameters_weight_"),
    ([32], "L_self_modules_conv1_modules_lin_neg_r_parameters_bias_"),
    ([32, 32], "L_self_modules_conv1_modules_lin_neg_r_parameters_weight_"),
    ([32, 32], "L_self_modules_conv1_modules_lin_pos_l_parameters_weight_"),
    ([32], "L_self_modules_conv1_modules_lin_pos_r_parameters_bias_"),
    ([32, 32], "L_self_modules_conv1_modules_lin_pos_r_parameters_weight_"),
    ([32, 64], "L_self_modules_convs_modules_0_modules_lin_neg_l_parameters_weight_"),
    ([32], "L_self_modules_convs_modules_0_modules_lin_neg_r_parameters_bias_"),
    ([32, 32], "L_self_modules_convs_modules_0_modules_lin_neg_r_parameters_weight_"),
    ([32, 64], "L_self_modules_convs_modules_0_modules_lin_pos_l_parameters_weight_"),
    ([32], "L_self_modules_convs_modules_0_modules_lin_pos_r_parameters_bias_"),
    ([32, 32], "L_self_modules_convs_modules_0_modules_lin_pos_r_parameters_weight_"),
    ([1000, 32], "L_x_"),
]
