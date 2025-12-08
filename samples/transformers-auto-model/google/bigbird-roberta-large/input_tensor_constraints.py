from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 17}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0, S1, 1], "L_logits_mask_"),
    (
        [4096],
        "L_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_bias_",
    ),
    (
        [4096, 1024],
        "L_self_modules_qa_classifier_modules_intermediate_modules_dense_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_qa_classifier_modules_output_modules_LayerNorm_parameters_bias_",
    ),
    (
        [1024],
        "L_self_modules_qa_classifier_modules_output_modules_LayerNorm_parameters_weight_",
    ),
    (
        [1024],
        "L_self_modules_qa_classifier_modules_output_modules_dense_parameters_bias_",
    ),
    (
        [1024, 4096],
        "L_self_modules_qa_classifier_modules_output_modules_dense_parameters_weight_",
    ),
    ([2], "L_self_modules_qa_classifier_modules_qa_outputs_parameters_bias_"),
    ([2, 1024], "L_self_modules_qa_classifier_modules_qa_outputs_parameters_weight_"),
    ([S0, S1, 1024], "dict_getitem_L_stack0_list_dict_keys_L_stack0_0_"),
]
