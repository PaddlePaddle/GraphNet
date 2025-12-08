from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1591, S1: 796}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([1, S0, 512], "L_pos_emb_"),
    ([512], "L_self_modules_after_norm_parameters_bias_"),
    ([512], "L_self_modules_after_norm_parameters_weight_"),
    (
        [2048],
        "L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_up_encoders_modules_0_modules_feed_forward_modules_w_2_parameters_weight_",
    ),
    ([512], "L_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_0_modules_norm_ff_parameters_weight_"),
    ([512], "L_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_0_modules_norm_mha_parameters_weight_"),
    (
        [512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_0_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [2048],
        "L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_up_encoders_modules_1_modules_feed_forward_modules_w_2_parameters_weight_",
    ),
    ([512], "L_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_1_modules_norm_ff_parameters_weight_"),
    ([512], "L_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_1_modules_norm_mha_parameters_weight_"),
    (
        [512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_1_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [2048],
        "L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_up_encoders_modules_2_modules_feed_forward_modules_w_2_parameters_weight_",
    ),
    ([512], "L_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_2_modules_norm_ff_parameters_weight_"),
    ([512], "L_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_2_modules_norm_mha_parameters_weight_"),
    (
        [512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_2_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [2048],
        "L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_1_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_self_modules_up_encoders_modules_3_modules_feed_forward_modules_w_2_parameters_weight_",
    ),
    ([512], "L_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_3_modules_norm_ff_parameters_weight_"),
    ([512], "L_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_bias_"),
    ([512], "L_self_modules_up_encoders_modules_3_modules_norm_mha_parameters_weight_"),
    (
        [512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_self_modules_up_encoders_modules_3_modules_self_attn_parameters_pos_bias_v_",
    ),
    ([1, 1, S1], "L_stack0_"),
    ([1, S1, 512], "L_xs_"),
]
