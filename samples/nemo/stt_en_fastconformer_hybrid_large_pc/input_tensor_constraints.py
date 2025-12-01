from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")

dynamic_dim_constraint_symbols = [S0, S1]

dynamic_dim_constraint_symbol2example_value = {S0: 1, S1: 521}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_"),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_"),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_"),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_"),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_"),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_"),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_"),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_16_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_16_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [512, 1, 9],
        "L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [1024],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [1024, 512, 1],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [512, 512, 1],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [2048],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [2048, 512],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [512, 2048],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_",
    ),
    ([512], "L_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_"),
    ([512], "L_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_"),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [512, 512],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [8, 64],
        "L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_",
    ),
    ([1, 9999, 512], "L_instance_modules_pos_enc_buffers_pe_"),
    ([256], "L_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_"),
    (
        [256, 1, 3, 3],
        "L_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_",
    ),
    ([256], "L_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_"),
    (
        [256, 1, 3, 3],
        "L_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_",
    ),
    ([256], "L_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_"),
    (
        [256, 256, 1, 1],
        "L_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_",
    ),
    ([256], "L_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_"),
    (
        [256, 1, 3, 3],
        "L_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_",
    ),
    ([256], "L_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_"),
    (
        [256, 256, 1, 1],
        "L_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_",
    ),
    ([512], "L_instance_modules_pre_encode_modules_out_parameters_bias_"),
    ([512, 2560], "L_instance_modules_pre_encode_modules_out_parameters_weight_"),
    ([S0, 80, S1], "L_kwargs_audio_signal_"),
    ([S0], "L_kwargs_length_"),
]
