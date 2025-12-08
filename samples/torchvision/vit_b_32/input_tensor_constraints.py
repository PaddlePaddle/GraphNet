from sympy import Symbol

S0 = Symbol("S0")

dynamic_dim_constraint_symbols = [S0]

dynamic_dim_constraint_symbol2example_value = {S0: 224}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([768], "L_self_modules_conv_proj_parameters_bias_"),
    ([768, 3, 32, 32], "L_self_modules_conv_proj_parameters_weight_"),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_0_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_10_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_11_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_1_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_2_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_3_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_4_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_5_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_6_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_7_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_8_modules_self_attention_parameters_in_proj_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_1_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_bias_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_ln_2_parameters_weight_",
    ),
    (
        [3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_bias_",
    ),
    (
        [3072, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_0_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_bias_",
    ),
    (
        [768, 3072],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_mlp_modules_3_parameters_weight_",
    ),
    (
        [768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_bias_",
    ),
    (
        [768, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_modules_out_proj_parameters_weight_",
    ),
    (
        [2304],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_bias_",
    ),
    (
        [2304, 768],
        "L_self_modules_encoder_modules_layers_modules_encoder_layer_9_modules_self_attention_parameters_in_proj_weight_",
    ),
    ([768], "L_self_modules_encoder_modules_ln_parameters_bias_"),
    ([768], "L_self_modules_encoder_modules_ln_parameters_weight_"),
    ([1, 50, 768], "L_self_modules_encoder_parameters_pos_embedding_"),
    ([1000], "L_self_modules_heads_modules_head_parameters_bias_"),
    ([1000, 768], "L_self_modules_heads_modules_head_parameters_weight_"),
    ([1, 1, 768], "L_self_parameters_class_token_"),
    ([1, 3, S0, S0], "L_x_"),
]
