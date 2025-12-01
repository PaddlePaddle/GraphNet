from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")
S2 = Symbol("S2")

dynamic_dim_constraint_symbols = [S0, S1, S2]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 10, S2: 1024}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [([S0, S1, S2], "L_features_token_embeddings_")]
