from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 2}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, 500], "L_edge_index_"),
    ([4], "L_self_buffers_alpha_"),
    ([1000, 64], "L_self_modules_embedding_parameters_weight_"),
]
