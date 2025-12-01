from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 6}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 7], "L_attention_mask_"),
    (
        [256],
        "L_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_0_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_0_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4, 256],
        "L_self_modules_bbox_embed_modules_0_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_1_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_1_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4, 256],
        "L_self_modules_bbox_embed_modules_1_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_2_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_2_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4, 256],
        "L_self_modules_bbox_embed_modules_2_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_3_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_3_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4, 256],
        "L_self_modules_bbox_embed_modules_3_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_4_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_4_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4, 256],
        "L_self_modules_bbox_embed_modules_4_modules_layers_modules_2_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_5_modules_layers_modules_0_parameters_weight_",
    ),
    (
        [256],
        "L_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_bias_",
    ),
    (
        [256, 256],
        "L_self_modules_bbox_embed_modules_5_modules_layers_modules_1_parameters_weight_",
    ),
    (
        [4],
        "L_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_bias_",
    ),
    (
        [4, 256],
        "L_self_modules_bbox_embed_modules_5_modules_layers_modules_2_parameters_weight_",
    ),
    ([1, 7, 256], "L_stack0_encoder_last_hidden_state_text"),
    ([1, 900, 4], "L_stack0_init_reference_points"),
    ([1, S0, 900, 256], "L_stack0_intermediate_hidden_states"),
    ([1, S0, 900, 4], "L_stack0_intermediate_reference_points"),
]
