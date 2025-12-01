from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 10}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0], "L_batch_"),
    ([994, 128], "L_self_modules_embedding_parameters_weight_"),
]
