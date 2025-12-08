from sympy import Symbol, Expr, Rel, Eq


dynamic_dim_constraint_symbols = []

dynamic_dim_constraint_symbol2example_value = {}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, 21], "L_attention_mask_"),
    ([1, 21, 16], "L_inputs_embeds_"),
    ([102, 16], "L_self_modules_embed_positions_parameters_weight_"),
    ([16], "L_self_modules_final_layer_norm_parameters_bias_"),
    ([16], "L_self_modules_final_layer_norm_parameters_weight_"),
    ([4], "L_self_modules_layers_modules_0_modules_fc1_parameters_bias_"),
    ([4, 16], "L_self_modules_layers_modules_0_modules_fc1_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_0_modules_fc2_parameters_bias_"),
    ([16, 4], "L_self_modules_layers_modules_0_modules_fc2_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_bias_"),
    (
        [16],
        "L_self_modules_layers_modules_0_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_0_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_0_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4], "L_self_modules_layers_modules_1_modules_fc1_parameters_bias_"),
    ([4, 16], "L_self_modules_layers_modules_1_modules_fc1_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_1_modules_fc2_parameters_bias_"),
    ([16, 4], "L_self_modules_layers_modules_1_modules_fc2_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_bias_"),
    (
        [16],
        "L_self_modules_layers_modules_1_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_1_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_1_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4], "L_self_modules_layers_modules_2_modules_fc1_parameters_bias_"),
    ([4, 16], "L_self_modules_layers_modules_2_modules_fc1_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_2_modules_fc2_parameters_bias_"),
    ([16, 4], "L_self_modules_layers_modules_2_modules_fc2_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_bias_"),
    (
        [16],
        "L_self_modules_layers_modules_2_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_2_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_2_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4], "L_self_modules_layers_modules_3_modules_fc1_parameters_bias_"),
    ([4, 16], "L_self_modules_layers_modules_3_modules_fc1_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_3_modules_fc2_parameters_bias_"),
    ([16, 4], "L_self_modules_layers_modules_3_modules_fc2_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_bias_"),
    (
        [16],
        "L_self_modules_layers_modules_3_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_3_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_3_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
    ([4], "L_self_modules_layers_modules_4_modules_fc1_parameters_bias_"),
    ([4, 16], "L_self_modules_layers_modules_4_modules_fc1_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_4_modules_fc2_parameters_bias_"),
    ([16, 4], "L_self_modules_layers_modules_4_modules_fc2_parameters_weight_"),
    ([16], "L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_bias_"),
    (
        [16],
        "L_self_modules_layers_modules_4_modules_final_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_bias_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_4_modules_self_attn_layer_norm_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_k_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_out_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_q_proj_parameters_weight_",
    ),
    (
        [16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_bias_",
    ),
    (
        [16, 16],
        "L_self_modules_layers_modules_4_modules_self_attn_modules_v_proj_parameters_weight_",
    ),
]
