from sympy import Symbol

S0 = Symbol("S0")
S1 = Symbol("S1")
S2 = Symbol("S2")

dynamic_dim_constraint_symbols = [S0, S1, S2]

dynamic_dim_constraint_symbol2example_value = {S0: 5000, S1: 1, S2: 521}

dynamic_dim_constraint_relations = []

dynamic_dim_constraint_input_shapes = [
    ([S0], "L_instance_buffers_seq_range_"),
    (
        [288],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_0_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_0_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_10_modules_conv_scale_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_conv_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_11_modules_conv_scale_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_conv_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_12_modules_conv_scale_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_conv_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_13_modules_conv_scale_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_conv_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_14_modules_conv_scale_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_conv_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_15_modules_conv_scale_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_conv_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_"),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_1_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_1_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_2_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_2_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_3_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_3_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_4_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_4_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_5_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_5_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_6_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_6_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_7_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_7_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_8_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_8_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_scale_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_",
    ),
    (
        [288, 1, 31],
        "L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_",
    ),
    (
        [288],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_",
    ),
    (
        [288, 144, 1],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_",
    ),
    (
        [144, 288, 1],
        "L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_",
    ),
    ([144], "L_instance_modules_layers_modules_9_modules_conv_scale_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_9_modules_conv_scale_parameters_scale_"),
    (
        [576],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_scale_",
    ),
    (
        [576],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_",
    ),
    (
        [576, 144],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_",
    ),
    (
        [144, 576],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_scale_",
    ),
    ([144], "L_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_"),
    ([144], "L_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_"),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_",
    ),
    (
        [144, 144],
        "L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_",
    ),
    (
        [4, 36],
        "L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_bias_",
    ),
    (
        [144],
        "L_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_scale_",
    ),
    ([1, 9999, 144], "L_instance_modules_pos_enc_buffers_pe_"),
    ([144], "L_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_"),
    (
        [144, 1, 3, 3],
        "L_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_",
    ),
    ([144], "L_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_"),
    (
        [144, 1, 3, 3],
        "L_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_",
    ),
    ([144], "L_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_"),
    (
        [144, 144, 1, 1],
        "L_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_",
    ),
    ([144], "L_instance_modules_pre_encode_modules_out_parameters_bias_"),
    ([144, 2880], "L_instance_modules_pre_encode_modules_out_parameters_weight_"),
    ([144], "L_instance_modules_pre_ln_parameters_bias_"),
    ([144], "L_instance_modules_pre_ln_parameters_weight_"),
    ([144], "L_instance_modules_time_recovery_layer_parameters_bias_"),
    ([144, 144], "L_instance_modules_time_recovery_layer_parameters_weight_"),
    ([144], "L_instance_modules_time_reduce_layer_modules_dw_conv_parameters_bias_"),
    (
        [144, 1, 5],
        "L_instance_modules_time_reduce_layer_modules_dw_conv_parameters_weight_",
    ),
    ([144], "L_instance_modules_time_reduce_layer_modules_pw_conv_parameters_bias_"),
    (
        [144, 144, 1],
        "L_instance_modules_time_reduce_layer_modules_pw_conv_parameters_weight_",
    ),
    ([1, 9999, 144], "L_instance_modules_time_reduce_pos_enc_buffers_pe_"),
    ([S1, 80, S2], "L_kwargs_audio_signal_"),
    ([S1], "L_kwargs_length_"),
]
