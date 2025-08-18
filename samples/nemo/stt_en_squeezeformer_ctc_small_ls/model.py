import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_kwargs_audio_signal_: torch.Tensor,
        L_kwargs_length_: torch.Tensor,
        L_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_pos_enc_buffers_pe_: torch.Tensor,
        L_instance_buffers_seq_range_: torch.Tensor,
        L_instance_modules_pre_ln_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_ln_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_time_reduce_layer_modules_dw_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_time_reduce_layer_modules_dw_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_time_reduce_layer_modules_pw_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_time_reduce_layer_modules_pw_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_time_reduce_pos_enc_buffers_pe_: torch.Tensor,
        L_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_time_recovery_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_time_recovery_layer_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_pos_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_u_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_v_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_mean_: torch.Tensor,
        L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_var_: torch.Tensor,
        L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_scale_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_kwargs_audio_signal_ = L_kwargs_audio_signal_
        l_kwargs_length_ = L_kwargs_length_
        l_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_ = (
            L_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_
        )
        l_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_ = (
            L_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_
        )
        l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_ = (
            L_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_
        )
        l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_ = (
            L_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_
        )
        l_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_ = (
            L_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_
        )
        l_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_ = (
            L_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_
        )
        l_instance_modules_pre_encode_modules_out_parameters_weight_ = (
            L_instance_modules_pre_encode_modules_out_parameters_weight_
        )
        l_instance_modules_pre_encode_modules_out_parameters_bias_ = (
            L_instance_modules_pre_encode_modules_out_parameters_bias_
        )
        l_instance_modules_pos_enc_buffers_pe_ = L_instance_modules_pos_enc_buffers_pe_
        l_instance_buffers_seq_range_ = L_instance_buffers_seq_range_
        l_instance_modules_pre_ln_parameters_weight_ = (
            L_instance_modules_pre_ln_parameters_weight_
        )
        l_instance_modules_pre_ln_parameters_bias_ = (
            L_instance_modules_pre_ln_parameters_bias_
        )
        l_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_0_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_0_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_0_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_0_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_1_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_1_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_1_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_1_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_2_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_2_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_2_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_2_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_3_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_3_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_3_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_3_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_4_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_4_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_4_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_4_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_5_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_5_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_5_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_5_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_6_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_6_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_6_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_6_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_7_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_7_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_7_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_7_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_time_reduce_layer_modules_dw_conv_parameters_weight_ = (
            L_instance_modules_time_reduce_layer_modules_dw_conv_parameters_weight_
        )
        l_instance_modules_time_reduce_layer_modules_dw_conv_parameters_bias_ = (
            L_instance_modules_time_reduce_layer_modules_dw_conv_parameters_bias_
        )
        l_instance_modules_time_reduce_layer_modules_pw_conv_parameters_weight_ = (
            L_instance_modules_time_reduce_layer_modules_pw_conv_parameters_weight_
        )
        l_instance_modules_time_reduce_layer_modules_pw_conv_parameters_bias_ = (
            L_instance_modules_time_reduce_layer_modules_pw_conv_parameters_bias_
        )
        l_instance_modules_time_reduce_pos_enc_buffers_pe_ = (
            L_instance_modules_time_reduce_pos_enc_buffers_pe_
        )
        l_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_8_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_8_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_8_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_8_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_ = (
            L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_ = (
            L_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_9_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_9_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_9_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_9_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_10_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_10_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_10_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_10_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_11_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_11_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_11_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_11_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_12_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_12_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_12_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_12_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_13_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_13_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_13_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_13_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_14_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_14_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_14_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_14_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_15_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_15_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_15_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_15_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_16_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_16_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_16_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_16_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_time_recovery_layer_parameters_weight_ = (
            L_instance_modules_time_recovery_layer_parameters_weight_
        )
        l_instance_modules_time_recovery_layer_parameters_bias_ = (
            L_instance_modules_time_recovery_layer_parameters_bias_
        )
        l_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_scale_ = L_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_scale_
        l_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_bias_ = L_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_bias_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_weight_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_weight_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_bias_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_bias_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_weight_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_weight_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_bias_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_bias_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_weight_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_weight_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_bias_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_bias_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_pos_parameters_weight_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_pos_parameters_weight_
        l_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_u_ = L_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_u_
        l_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_v_ = L_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_v_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_weight_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_weight_
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_bias_ = L_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_bias_
        l_instance_modules_layers_modules_17_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_17_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_17_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_17_modules_norm_self_att_parameters_bias_
        )
        l_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_scale_ = L_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_scale_
        l_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_bias_ = L_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_bias_
        l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_17_modules_conv_scale_parameters_scale_ = (
            L_instance_modules_layers_modules_17_modules_conv_scale_parameters_scale_
        )
        l_instance_modules_layers_modules_17_modules_conv_scale_parameters_bias_ = (
            L_instance_modules_layers_modules_17_modules_conv_scale_parameters_bias_
        )
        l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_weight_ = L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_weight_
        l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_bias_ = L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_bias_
        l_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_weight_ = L_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_weight_
        l_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_bias_ = L_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_bias_
        l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_mean_ = L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_mean_
        l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_var_ = L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_var_
        l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_weight_ = L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_weight_
        l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_bias_ = L_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_bias_
        l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_weight_ = L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_weight_
        l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_bias_ = L_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_bias_
        l_instance_modules_layers_modules_17_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_17_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_17_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_17_modules_norm_conv_parameters_bias_
        )
        l_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_scale_ = L_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_scale_
        l_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_bias_ = L_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_bias_
        l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_bias_
        audio_signal = torch.transpose(l_kwargs_audio_signal_, 1, 2)
        l_kwargs_audio_signal_ = None
        to = l_kwargs_length_.to(dtype=torch.float32)
        l_kwargs_length_ = None
        add = to + -1
        to = None
        div = torch.div(add, 2)
        add = None
        lengths = div + 1.0
        div = None
        lengths_1 = torch.floor(lengths)
        lengths = None
        to_1 = lengths_1.to(dtype=torch.float32)
        lengths_1 = None
        add_2 = to_1 + -1
        to_1 = None
        div_1 = torch.div(add_2, 2)
        add_2 = None
        lengths_2 = div_1 + 1.0
        div_1 = None
        lengths_3 = torch.floor(lengths_2)
        lengths_2 = None
        lengths_4 = lengths_3.to(dtype=torch.int32)
        lengths_3 = None
        x = audio_signal.unsqueeze(1)
        audio_signal = None
        input_1 = torch.conv2d(
            x,
            l_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x = (
            l_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_ = None
        input_2 = torch.nn.functional.relu(input_1, inplace=False)
        input_1 = None
        input_3 = torch.conv2d(
            input_2,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            196,
        )
        input_2 = (
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_ = None
        input_4 = torch.conv2d(
            input_3,
            l_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = (
            l_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(input_4, inplace=False)
        input_4 = None
        transpose_1 = input_5.transpose(1, 2)
        input_5 = None
        reshape = transpose_1.reshape(1, 131, -1)
        transpose_1 = None
        x_1 = torch._C._nn.linear(
            reshape,
            l_instance_modules_pre_encode_modules_out_parameters_weight_,
            l_instance_modules_pre_encode_modules_out_parameters_bias_,
        )
        reshape = (
            l_instance_modules_pre_encode_modules_out_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_out_parameters_bias_ = None
        x_2 = x_1 * 14.0
        x_1 = None
        pos_emb = l_instance_modules_pos_enc_buffers_pe_[
            (slice(None, None, None), slice(4869, 5130, None))
        ]
        l_instance_modules_pos_enc_buffers_pe_ = None
        audio_signal_1 = torch.nn.functional.dropout(x_2, 0.1, False, False)
        x_2 = None
        getitem_1 = l_instance_buffers_seq_range_[slice(None, 131, None)]
        l_instance_buffers_seq_range_ = None
        expand = getitem_1.expand(1, -1)
        getitem_1 = None
        unsqueeze_1 = lengths_4.unsqueeze(-1)
        mask = expand < unsqueeze_1
        expand = unsqueeze_1 = None
        unsqueeze_2 = mask.unsqueeze(1)
        att_mask = unsqueeze_2.repeat([1, 131, 1])
        unsqueeze_2 = None
        transpose_2 = att_mask.transpose(1, 2)
        att_mask_1 = torch.logical_and(att_mask, transpose_2)
        att_mask = transpose_2 = None
        att_mask_2 = ~att_mask_1
        att_mask_1 = None
        pad_mask = ~mask
        mask = None
        audio_signal_2 = torch.nn.functional.layer_norm(
            audio_signal_1,
            (196,),
            l_instance_modules_pre_ln_parameters_weight_,
            l_instance_modules_pre_ln_parameters_bias_,
            1e-05,
        )
        audio_signal_1 = (
            l_instance_modules_pre_ln_parameters_weight_
        ) = l_instance_modules_pre_ln_parameters_bias_ = None
        scale = l_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias = l_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_1 = audio_signal_2 * scale
        scale = None
        x_3 = mul_1 + bias
        mul_1 = bias = None
        linear_1 = torch._C._nn.linear(
            x_3,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q = linear_1.view(1, -1, 4, 49)
        linear_1 = None
        linear_2 = torch._C._nn.linear(
            x_3,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k = linear_2.view(1, -1, 4, 49)
        linear_2 = None
        linear_3 = torch._C._nn.linear(
            x_3,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_3 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v = linear_3.view(1, -1, 4, 49)
        linear_3 = None
        q_1 = q.transpose(1, 2)
        q = None
        k_1 = k.transpose(1, 2)
        k = None
        v_1 = v.transpose(1, 2)
        v = None
        q_2 = q_1.transpose(1, 2)
        q_1 = None
        linear_4 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p = linear_4.view(1, -1, 4, 49)
        linear_4 = None
        p_1 = p.transpose(1, 2)
        p = None
        add_5 = (
            q_2
            + l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u = add_5.transpose(1, 2)
        add_5 = None
        add_6 = (
            q_2
            + l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_
        )
        q_2 = (
            l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v = add_6.transpose(1, 2)
        add_6 = None
        transpose_10 = p_1.transpose(-2, -1)
        p_1 = None
        matrix_bd = torch.matmul(q_with_bias_v, transpose_10)
        q_with_bias_v = transpose_10 = None
        x_4 = torch._C._nn.pad(matrix_bd, (1, 0), "constant", None)
        matrix_bd = None
        x_5 = x_4.view(1, 4, -1, 131)
        x_4 = None
        getitem_2 = x_5[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_5 = None
        x_6 = getitem_2.view(1, 4, 131, 261)
        getitem_2 = None
        transpose_11 = k_1.transpose(-2, -1)
        k_1 = None
        matrix_ac = torch.matmul(q_with_bias_u, transpose_11)
        q_with_bias_u = transpose_11 = None
        matrix_bd_1 = x_6[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_6 = None
        add_7 = matrix_ac + matrix_bd_1
        matrix_ac = matrix_bd_1 = None
        scores = add_7 / 7.0
        add_7 = None
        mask_1 = att_mask_2.unsqueeze(1)
        scores_1 = scores.masked_fill(mask_1, -10000.0)
        scores = None
        softmax = torch.softmax(scores_1, dim=-1)
        scores_1 = None
        attn = softmax.masked_fill(mask_1, 0.0)
        softmax = mask_1 = None
        p_attn = torch.nn.functional.dropout(attn, 0.1, False, False)
        attn = None
        x_7 = torch.matmul(p_attn, v_1)
        p_attn = v_1 = None
        transpose_12 = x_7.transpose(1, 2)
        x_7 = None
        x_8 = transpose_12.reshape(1, -1, 196)
        transpose_12 = None
        out = torch._C._nn.linear(
            x_8,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_8 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_2 = torch.nn.functional.dropout(out, 0.1, False, False)
        out = None
        x_9 = audio_signal_2 + dropout_2
        audio_signal_2 = dropout_2 = None
        x_10 = torch.nn.functional.layer_norm(
            x_9,
            (196,),
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_9 = (
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_
        ) = None
        scale_1 = l_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_1 = l_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_2 = x_10 * scale_1
        scale_1 = None
        x_11 = mul_2 + bias_1
        mul_2 = bias_1 = None
        x_12 = torch._C._nn.linear(
            x_11,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_11 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_13 = torch.nn.functional.silu(x_12, inplace=False)
        x_12 = None
        x_14 = torch.nn.functional.dropout(x_13, 0.1, False, False)
        x_13 = None
        x_15 = torch._C._nn.linear(
            x_14,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_14 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_4 = torch.nn.functional.dropout(x_15, 0.1, False, False)
        x_15 = None
        mul_3 = dropout_4 * 1.0
        dropout_4 = None
        x_16 = x_10 + mul_3
        x_10 = mul_3 = None
        x_17 = torch.nn.functional.layer_norm(
            x_16,
            (196,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_16 = l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_2 = l_instance_modules_layers_modules_0_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_conv_scale_parameters_scale_ = None
        bias_2 = l_instance_modules_layers_modules_0_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_conv_scale_parameters_bias_ = None
        mul_4 = x_17 * scale_2
        scale_2 = None
        x_18 = mul_4 + bias_2
        mul_4 = bias_2 = None
        x_19 = x_18.transpose(1, 2)
        x_18 = None
        x_20 = torch.conv1d(
            x_19,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_19 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_21 = torch.nn.functional.silu(x_20, inplace=True)
        x_20 = None
        unsqueeze_4 = pad_mask.unsqueeze(1)
        x_22 = x_21.masked_fill(unsqueeze_4, 0.0)
        x_21 = unsqueeze_4 = None
        new_x = torch._C._nn.pad(x_22, (15, 15), "constant", None)
        x_22 = None
        x_23 = torch.conv1d(
            new_x,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_25 = torch.nn.functional.silu(x_24, inplace=False)
        x_24 = None
        x_26 = torch.conv1d(
            x_25,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_25 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_27 = x_26.transpose(1, 2)
        x_26 = None
        dropout_5 = torch.nn.functional.dropout(x_27, 0.1, False, False)
        x_27 = None
        x_28 = x_17 + dropout_5
        x_17 = dropout_5 = None
        x_29 = torch.nn.functional.layer_norm(
            x_28,
            (196,),
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_28 = (
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_
        ) = None
        scale_3 = l_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_3 = l_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_0_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_5 = x_29 * scale_3
        scale_3 = None
        x_30 = mul_5 + bias_3
        mul_5 = bias_3 = None
        x_31 = torch._C._nn.linear(
            x_30,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_30 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_32 = torch.nn.functional.silu(x_31, inplace=False)
        x_31 = None
        x_33 = torch.nn.functional.dropout(x_32, 0.1, False, False)
        x_32 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_33 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_7 = torch.nn.functional.dropout(x_34, 0.1, False, False)
        x_34 = None
        mul_6 = dropout_7 * 1.0
        dropout_7 = None
        x_35 = x_29 + mul_6
        x_29 = mul_6 = None
        x_36 = torch.nn.functional.layer_norm(
            x_35,
            (196,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_35 = l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_4 = l_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_4 = l_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_7 = x_36 * scale_4
        scale_4 = None
        x_37 = mul_7 + bias_4
        mul_7 = bias_4 = None
        linear_10 = torch._C._nn.linear(
            x_37,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_3 = linear_10.view(1, -1, 4, 49)
        linear_10 = None
        linear_11 = torch._C._nn.linear(
            x_37,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_2 = linear_11.view(1, -1, 4, 49)
        linear_11 = None
        linear_12 = torch._C._nn.linear(
            x_37,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_37 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_2 = linear_12.view(1, -1, 4, 49)
        linear_12 = None
        q_4 = q_3.transpose(1, 2)
        q_3 = None
        k_3 = k_2.transpose(1, 2)
        k_2 = None
        v_3 = v_2.transpose(1, 2)
        v_2 = None
        q_5 = q_4.transpose(1, 2)
        q_4 = None
        linear_13 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_2 = linear_13.view(1, -1, 4, 49)
        linear_13 = None
        p_3 = p_2.transpose(1, 2)
        p_2 = None
        add_16 = (
            q_5
            + l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_1 = add_16.transpose(1, 2)
        add_16 = None
        add_17 = (
            q_5
            + l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_
        )
        q_5 = (
            l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_1 = add_17.transpose(1, 2)
        add_17 = None
        transpose_22 = p_3.transpose(-2, -1)
        p_3 = None
        matrix_bd_2 = torch.matmul(q_with_bias_v_1, transpose_22)
        q_with_bias_v_1 = transpose_22 = None
        x_38 = torch._C._nn.pad(matrix_bd_2, (1, 0), "constant", None)
        matrix_bd_2 = None
        x_39 = x_38.view(1, 4, -1, 131)
        x_38 = None
        getitem_4 = x_39[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_39 = None
        x_40 = getitem_4.view(1, 4, 131, 261)
        getitem_4 = None
        transpose_23 = k_3.transpose(-2, -1)
        k_3 = None
        matrix_ac_1 = torch.matmul(q_with_bias_u_1, transpose_23)
        q_with_bias_u_1 = transpose_23 = None
        matrix_bd_3 = x_40[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_40 = None
        add_18 = matrix_ac_1 + matrix_bd_3
        matrix_ac_1 = matrix_bd_3 = None
        scores_2 = add_18 / 7.0
        add_18 = None
        mask_2 = att_mask_2.unsqueeze(1)
        scores_3 = scores_2.masked_fill(mask_2, -10000.0)
        scores_2 = None
        softmax_1 = torch.softmax(scores_3, dim=-1)
        scores_3 = None
        attn_1 = softmax_1.masked_fill(mask_2, 0.0)
        softmax_1 = mask_2 = None
        p_attn_1 = torch.nn.functional.dropout(attn_1, 0.1, False, False)
        attn_1 = None
        x_41 = torch.matmul(p_attn_1, v_3)
        p_attn_1 = v_3 = None
        transpose_24 = x_41.transpose(1, 2)
        x_41 = None
        x_42 = transpose_24.reshape(1, -1, 196)
        transpose_24 = None
        out_1 = torch._C._nn.linear(
            x_42,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_42 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_9 = torch.nn.functional.dropout(out_1, 0.1, False, False)
        out_1 = None
        x_43 = x_36 + dropout_9
        x_36 = dropout_9 = None
        x_44 = torch.nn.functional.layer_norm(
            x_43,
            (196,),
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_43 = (
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_
        ) = None
        scale_5 = l_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_5 = l_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_8 = x_44 * scale_5
        scale_5 = None
        x_45 = mul_8 + bias_5
        mul_8 = bias_5 = None
        x_46 = torch._C._nn.linear(
            x_45,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_45 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_47 = torch.nn.functional.silu(x_46, inplace=False)
        x_46 = None
        x_48 = torch.nn.functional.dropout(x_47, 0.1, False, False)
        x_47 = None
        x_49 = torch._C._nn.linear(
            x_48,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_48 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_11 = torch.nn.functional.dropout(x_49, 0.1, False, False)
        x_49 = None
        mul_9 = dropout_11 * 1.0
        dropout_11 = None
        x_50 = x_44 + mul_9
        x_44 = mul_9 = None
        x_51 = torch.nn.functional.layer_norm(
            x_50,
            (196,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_50 = l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_6 = l_instance_modules_layers_modules_1_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_conv_scale_parameters_scale_ = None
        bias_6 = l_instance_modules_layers_modules_1_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_conv_scale_parameters_bias_ = None
        mul_10 = x_51 * scale_6
        scale_6 = None
        x_52 = mul_10 + bias_6
        mul_10 = bias_6 = None
        x_53 = x_52.transpose(1, 2)
        x_52 = None
        x_54 = torch.conv1d(
            x_53,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_53 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_55 = torch.nn.functional.silu(x_54, inplace=True)
        x_54 = None
        unsqueeze_6 = pad_mask.unsqueeze(1)
        x_56 = x_55.masked_fill(unsqueeze_6, 0.0)
        x_55 = unsqueeze_6 = None
        new_x_1 = torch._C._nn.pad(x_56, (15, 15), "constant", None)
        x_56 = None
        x_57 = torch.conv1d(
            new_x_1,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_1 = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_59 = torch.nn.functional.silu(x_58, inplace=False)
        x_58 = None
        x_60 = torch.conv1d(
            x_59,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_59 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_61 = x_60.transpose(1, 2)
        x_60 = None
        dropout_12 = torch.nn.functional.dropout(x_61, 0.1, False, False)
        x_61 = None
        x_62 = x_51 + dropout_12
        x_51 = dropout_12 = None
        x_63 = torch.nn.functional.layer_norm(
            x_62,
            (196,),
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_62 = (
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_
        ) = None
        scale_7 = l_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_7 = l_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_1_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_11 = x_63 * scale_7
        scale_7 = None
        x_64 = mul_11 + bias_7
        mul_11 = bias_7 = None
        x_65 = torch._C._nn.linear(
            x_64,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_64 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_66 = torch.nn.functional.silu(x_65, inplace=False)
        x_65 = None
        x_67 = torch.nn.functional.dropout(x_66, 0.1, False, False)
        x_66 = None
        x_68 = torch._C._nn.linear(
            x_67,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_67 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_14 = torch.nn.functional.dropout(x_68, 0.1, False, False)
        x_68 = None
        mul_12 = dropout_14 * 1.0
        dropout_14 = None
        x_69 = x_63 + mul_12
        x_63 = mul_12 = None
        x_70 = torch.nn.functional.layer_norm(
            x_69,
            (196,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_69 = l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_8 = l_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_8 = l_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_13 = x_70 * scale_8
        scale_8 = None
        x_71 = mul_13 + bias_8
        mul_13 = bias_8 = None
        linear_19 = torch._C._nn.linear(
            x_71,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_6 = linear_19.view(1, -1, 4, 49)
        linear_19 = None
        linear_20 = torch._C._nn.linear(
            x_71,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_4 = linear_20.view(1, -1, 4, 49)
        linear_20 = None
        linear_21 = torch._C._nn.linear(
            x_71,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_71 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_4 = linear_21.view(1, -1, 4, 49)
        linear_21 = None
        q_7 = q_6.transpose(1, 2)
        q_6 = None
        k_5 = k_4.transpose(1, 2)
        k_4 = None
        v_5 = v_4.transpose(1, 2)
        v_4 = None
        q_8 = q_7.transpose(1, 2)
        q_7 = None
        linear_22 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_4 = linear_22.view(1, -1, 4, 49)
        linear_22 = None
        p_5 = p_4.transpose(1, 2)
        p_4 = None
        add_27 = (
            q_8
            + l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_2 = add_27.transpose(1, 2)
        add_27 = None
        add_28 = (
            q_8
            + l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_
        )
        q_8 = (
            l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_2 = add_28.transpose(1, 2)
        add_28 = None
        transpose_34 = p_5.transpose(-2, -1)
        p_5 = None
        matrix_bd_4 = torch.matmul(q_with_bias_v_2, transpose_34)
        q_with_bias_v_2 = transpose_34 = None
        x_72 = torch._C._nn.pad(matrix_bd_4, (1, 0), "constant", None)
        matrix_bd_4 = None
        x_73 = x_72.view(1, 4, -1, 131)
        x_72 = None
        getitem_6 = x_73[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_73 = None
        x_74 = getitem_6.view(1, 4, 131, 261)
        getitem_6 = None
        transpose_35 = k_5.transpose(-2, -1)
        k_5 = None
        matrix_ac_2 = torch.matmul(q_with_bias_u_2, transpose_35)
        q_with_bias_u_2 = transpose_35 = None
        matrix_bd_5 = x_74[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_74 = None
        add_29 = matrix_ac_2 + matrix_bd_5
        matrix_ac_2 = matrix_bd_5 = None
        scores_4 = add_29 / 7.0
        add_29 = None
        mask_3 = att_mask_2.unsqueeze(1)
        scores_5 = scores_4.masked_fill(mask_3, -10000.0)
        scores_4 = None
        softmax_2 = torch.softmax(scores_5, dim=-1)
        scores_5 = None
        attn_2 = softmax_2.masked_fill(mask_3, 0.0)
        softmax_2 = mask_3 = None
        p_attn_2 = torch.nn.functional.dropout(attn_2, 0.1, False, False)
        attn_2 = None
        x_75 = torch.matmul(p_attn_2, v_5)
        p_attn_2 = v_5 = None
        transpose_36 = x_75.transpose(1, 2)
        x_75 = None
        x_76 = transpose_36.reshape(1, -1, 196)
        transpose_36 = None
        out_2 = torch._C._nn.linear(
            x_76,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_76 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_16 = torch.nn.functional.dropout(out_2, 0.1, False, False)
        out_2 = None
        x_77 = x_70 + dropout_16
        x_70 = dropout_16 = None
        x_78 = torch.nn.functional.layer_norm(
            x_77,
            (196,),
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_77 = (
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_
        ) = None
        scale_9 = l_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_9 = l_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_14 = x_78 * scale_9
        scale_9 = None
        x_79 = mul_14 + bias_9
        mul_14 = bias_9 = None
        x_80 = torch._C._nn.linear(
            x_79,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_79 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_81 = torch.nn.functional.silu(x_80, inplace=False)
        x_80 = None
        x_82 = torch.nn.functional.dropout(x_81, 0.1, False, False)
        x_81 = None
        x_83 = torch._C._nn.linear(
            x_82,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_82 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_18 = torch.nn.functional.dropout(x_83, 0.1, False, False)
        x_83 = None
        mul_15 = dropout_18 * 1.0
        dropout_18 = None
        x_84 = x_78 + mul_15
        x_78 = mul_15 = None
        x_85 = torch.nn.functional.layer_norm(
            x_84,
            (196,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_84 = l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_10 = l_instance_modules_layers_modules_2_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_conv_scale_parameters_scale_ = None
        bias_10 = l_instance_modules_layers_modules_2_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_conv_scale_parameters_bias_ = None
        mul_16 = x_85 * scale_10
        scale_10 = None
        x_86 = mul_16 + bias_10
        mul_16 = bias_10 = None
        x_87 = x_86.transpose(1, 2)
        x_86 = None
        x_88 = torch.conv1d(
            x_87,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_87 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_89 = torch.nn.functional.silu(x_88, inplace=True)
        x_88 = None
        unsqueeze_8 = pad_mask.unsqueeze(1)
        x_90 = x_89.masked_fill(unsqueeze_8, 0.0)
        x_89 = unsqueeze_8 = None
        new_x_2 = torch._C._nn.pad(x_90, (15, 15), "constant", None)
        x_90 = None
        x_91 = torch.conv1d(
            new_x_2,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_2 = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_93 = torch.nn.functional.silu(x_92, inplace=False)
        x_92 = None
        x_94 = torch.conv1d(
            x_93,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_93 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_95 = x_94.transpose(1, 2)
        x_94 = None
        dropout_19 = torch.nn.functional.dropout(x_95, 0.1, False, False)
        x_95 = None
        x_96 = x_85 + dropout_19
        x_85 = dropout_19 = None
        x_97 = torch.nn.functional.layer_norm(
            x_96,
            (196,),
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_96 = (
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_
        ) = None
        scale_11 = l_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_11 = l_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_2_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_17 = x_97 * scale_11
        scale_11 = None
        x_98 = mul_17 + bias_11
        mul_17 = bias_11 = None
        x_99 = torch._C._nn.linear(
            x_98,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_98 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_100 = torch.nn.functional.silu(x_99, inplace=False)
        x_99 = None
        x_101 = torch.nn.functional.dropout(x_100, 0.1, False, False)
        x_100 = None
        x_102 = torch._C._nn.linear(
            x_101,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_101 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_21 = torch.nn.functional.dropout(x_102, 0.1, False, False)
        x_102 = None
        mul_18 = dropout_21 * 1.0
        dropout_21 = None
        x_103 = x_97 + mul_18
        x_97 = mul_18 = None
        x_104 = torch.nn.functional.layer_norm(
            x_103,
            (196,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_103 = l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_12 = l_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_12 = l_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_19 = x_104 * scale_12
        scale_12 = None
        x_105 = mul_19 + bias_12
        mul_19 = bias_12 = None
        linear_28 = torch._C._nn.linear(
            x_105,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_9 = linear_28.view(1, -1, 4, 49)
        linear_28 = None
        linear_29 = torch._C._nn.linear(
            x_105,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_6 = linear_29.view(1, -1, 4, 49)
        linear_29 = None
        linear_30 = torch._C._nn.linear(
            x_105,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_105 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_6 = linear_30.view(1, -1, 4, 49)
        linear_30 = None
        q_10 = q_9.transpose(1, 2)
        q_9 = None
        k_7 = k_6.transpose(1, 2)
        k_6 = None
        v_7 = v_6.transpose(1, 2)
        v_6 = None
        q_11 = q_10.transpose(1, 2)
        q_10 = None
        linear_31 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_6 = linear_31.view(1, -1, 4, 49)
        linear_31 = None
        p_7 = p_6.transpose(1, 2)
        p_6 = None
        add_38 = (
            q_11
            + l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_3 = add_38.transpose(1, 2)
        add_38 = None
        add_39 = (
            q_11
            + l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_
        )
        q_11 = (
            l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_3 = add_39.transpose(1, 2)
        add_39 = None
        transpose_46 = p_7.transpose(-2, -1)
        p_7 = None
        matrix_bd_6 = torch.matmul(q_with_bias_v_3, transpose_46)
        q_with_bias_v_3 = transpose_46 = None
        x_106 = torch._C._nn.pad(matrix_bd_6, (1, 0), "constant", None)
        matrix_bd_6 = None
        x_107 = x_106.view(1, 4, -1, 131)
        x_106 = None
        getitem_8 = x_107[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_107 = None
        x_108 = getitem_8.view(1, 4, 131, 261)
        getitem_8 = None
        transpose_47 = k_7.transpose(-2, -1)
        k_7 = None
        matrix_ac_3 = torch.matmul(q_with_bias_u_3, transpose_47)
        q_with_bias_u_3 = transpose_47 = None
        matrix_bd_7 = x_108[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_108 = None
        add_40 = matrix_ac_3 + matrix_bd_7
        matrix_ac_3 = matrix_bd_7 = None
        scores_6 = add_40 / 7.0
        add_40 = None
        mask_4 = att_mask_2.unsqueeze(1)
        scores_7 = scores_6.masked_fill(mask_4, -10000.0)
        scores_6 = None
        softmax_3 = torch.softmax(scores_7, dim=-1)
        scores_7 = None
        attn_3 = softmax_3.masked_fill(mask_4, 0.0)
        softmax_3 = mask_4 = None
        p_attn_3 = torch.nn.functional.dropout(attn_3, 0.1, False, False)
        attn_3 = None
        x_109 = torch.matmul(p_attn_3, v_7)
        p_attn_3 = v_7 = None
        transpose_48 = x_109.transpose(1, 2)
        x_109 = None
        x_110 = transpose_48.reshape(1, -1, 196)
        transpose_48 = None
        out_3 = torch._C._nn.linear(
            x_110,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_110 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_23 = torch.nn.functional.dropout(out_3, 0.1, False, False)
        out_3 = None
        x_111 = x_104 + dropout_23
        x_104 = dropout_23 = None
        x_112 = torch.nn.functional.layer_norm(
            x_111,
            (196,),
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_111 = (
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_
        ) = None
        scale_13 = l_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_13 = l_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_20 = x_112 * scale_13
        scale_13 = None
        x_113 = mul_20 + bias_13
        mul_20 = bias_13 = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_113 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_115 = torch.nn.functional.silu(x_114, inplace=False)
        x_114 = None
        x_116 = torch.nn.functional.dropout(x_115, 0.1, False, False)
        x_115 = None
        x_117 = torch._C._nn.linear(
            x_116,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_116 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_25 = torch.nn.functional.dropout(x_117, 0.1, False, False)
        x_117 = None
        mul_21 = dropout_25 * 1.0
        dropout_25 = None
        x_118 = x_112 + mul_21
        x_112 = mul_21 = None
        x_119 = torch.nn.functional.layer_norm(
            x_118,
            (196,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_118 = l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_14 = l_instance_modules_layers_modules_3_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_conv_scale_parameters_scale_ = None
        bias_14 = l_instance_modules_layers_modules_3_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_conv_scale_parameters_bias_ = None
        mul_22 = x_119 * scale_14
        scale_14 = None
        x_120 = mul_22 + bias_14
        mul_22 = bias_14 = None
        x_121 = x_120.transpose(1, 2)
        x_120 = None
        x_122 = torch.conv1d(
            x_121,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_121 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_123 = torch.nn.functional.silu(x_122, inplace=True)
        x_122 = None
        unsqueeze_10 = pad_mask.unsqueeze(1)
        x_124 = x_123.masked_fill(unsqueeze_10, 0.0)
        x_123 = unsqueeze_10 = None
        new_x_3 = torch._C._nn.pad(x_124, (15, 15), "constant", None)
        x_124 = None
        x_125 = torch.conv1d(
            new_x_3,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_3 = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_127 = torch.nn.functional.silu(x_126, inplace=False)
        x_126 = None
        x_128 = torch.conv1d(
            x_127,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_127 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_129 = x_128.transpose(1, 2)
        x_128 = None
        dropout_26 = torch.nn.functional.dropout(x_129, 0.1, False, False)
        x_129 = None
        x_130 = x_119 + dropout_26
        x_119 = dropout_26 = None
        x_131 = torch.nn.functional.layer_norm(
            x_130,
            (196,),
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_130 = (
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_
        ) = None
        scale_15 = l_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_15 = l_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_3_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_23 = x_131 * scale_15
        scale_15 = None
        x_132 = mul_23 + bias_15
        mul_23 = bias_15 = None
        x_133 = torch._C._nn.linear(
            x_132,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_132 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_134 = torch.nn.functional.silu(x_133, inplace=False)
        x_133 = None
        x_135 = torch.nn.functional.dropout(x_134, 0.1, False, False)
        x_134 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_135 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_28 = torch.nn.functional.dropout(x_136, 0.1, False, False)
        x_136 = None
        mul_24 = dropout_28 * 1.0
        dropout_28 = None
        x_137 = x_131 + mul_24
        x_131 = mul_24 = None
        x_138 = torch.nn.functional.layer_norm(
            x_137,
            (196,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_137 = l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_16 = l_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_16 = l_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_25 = x_138 * scale_16
        scale_16 = None
        x_139 = mul_25 + bias_16
        mul_25 = bias_16 = None
        linear_37 = torch._C._nn.linear(
            x_139,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_12 = linear_37.view(1, -1, 4, 49)
        linear_37 = None
        linear_38 = torch._C._nn.linear(
            x_139,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_8 = linear_38.view(1, -1, 4, 49)
        linear_38 = None
        linear_39 = torch._C._nn.linear(
            x_139,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_139 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_8 = linear_39.view(1, -1, 4, 49)
        linear_39 = None
        q_13 = q_12.transpose(1, 2)
        q_12 = None
        k_9 = k_8.transpose(1, 2)
        k_8 = None
        v_9 = v_8.transpose(1, 2)
        v_8 = None
        q_14 = q_13.transpose(1, 2)
        q_13 = None
        linear_40 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_8 = linear_40.view(1, -1, 4, 49)
        linear_40 = None
        p_9 = p_8.transpose(1, 2)
        p_8 = None
        add_49 = (
            q_14
            + l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_4 = add_49.transpose(1, 2)
        add_49 = None
        add_50 = (
            q_14
            + l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_
        )
        q_14 = (
            l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_4 = add_50.transpose(1, 2)
        add_50 = None
        transpose_58 = p_9.transpose(-2, -1)
        p_9 = None
        matrix_bd_8 = torch.matmul(q_with_bias_v_4, transpose_58)
        q_with_bias_v_4 = transpose_58 = None
        x_140 = torch._C._nn.pad(matrix_bd_8, (1, 0), "constant", None)
        matrix_bd_8 = None
        x_141 = x_140.view(1, 4, -1, 131)
        x_140 = None
        getitem_10 = x_141[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_141 = None
        x_142 = getitem_10.view(1, 4, 131, 261)
        getitem_10 = None
        transpose_59 = k_9.transpose(-2, -1)
        k_9 = None
        matrix_ac_4 = torch.matmul(q_with_bias_u_4, transpose_59)
        q_with_bias_u_4 = transpose_59 = None
        matrix_bd_9 = x_142[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_142 = None
        add_51 = matrix_ac_4 + matrix_bd_9
        matrix_ac_4 = matrix_bd_9 = None
        scores_8 = add_51 / 7.0
        add_51 = None
        mask_5 = att_mask_2.unsqueeze(1)
        scores_9 = scores_8.masked_fill(mask_5, -10000.0)
        scores_8 = None
        softmax_4 = torch.softmax(scores_9, dim=-1)
        scores_9 = None
        attn_4 = softmax_4.masked_fill(mask_5, 0.0)
        softmax_4 = mask_5 = None
        p_attn_4 = torch.nn.functional.dropout(attn_4, 0.1, False, False)
        attn_4 = None
        x_143 = torch.matmul(p_attn_4, v_9)
        p_attn_4 = v_9 = None
        transpose_60 = x_143.transpose(1, 2)
        x_143 = None
        x_144 = transpose_60.reshape(1, -1, 196)
        transpose_60 = None
        out_4 = torch._C._nn.linear(
            x_144,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_144 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_30 = torch.nn.functional.dropout(out_4, 0.1, False, False)
        out_4 = None
        x_145 = x_138 + dropout_30
        x_138 = dropout_30 = None
        x_146 = torch.nn.functional.layer_norm(
            x_145,
            (196,),
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_145 = (
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_
        ) = None
        scale_17 = l_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_17 = l_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_26 = x_146 * scale_17
        scale_17 = None
        x_147 = mul_26 + bias_17
        mul_26 = bias_17 = None
        x_148 = torch._C._nn.linear(
            x_147,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_147 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_149 = torch.nn.functional.silu(x_148, inplace=False)
        x_148 = None
        x_150 = torch.nn.functional.dropout(x_149, 0.1, False, False)
        x_149 = None
        x_151 = torch._C._nn.linear(
            x_150,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_150 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_32 = torch.nn.functional.dropout(x_151, 0.1, False, False)
        x_151 = None
        mul_27 = dropout_32 * 1.0
        dropout_32 = None
        x_152 = x_146 + mul_27
        x_146 = mul_27 = None
        x_153 = torch.nn.functional.layer_norm(
            x_152,
            (196,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_152 = l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_18 = l_instance_modules_layers_modules_4_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_conv_scale_parameters_scale_ = None
        bias_18 = l_instance_modules_layers_modules_4_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_conv_scale_parameters_bias_ = None
        mul_28 = x_153 * scale_18
        scale_18 = None
        x_154 = mul_28 + bias_18
        mul_28 = bias_18 = None
        x_155 = x_154.transpose(1, 2)
        x_154 = None
        x_156 = torch.conv1d(
            x_155,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_155 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_157 = torch.nn.functional.silu(x_156, inplace=True)
        x_156 = None
        unsqueeze_12 = pad_mask.unsqueeze(1)
        x_158 = x_157.masked_fill(unsqueeze_12, 0.0)
        x_157 = unsqueeze_12 = None
        new_x_4 = torch._C._nn.pad(x_158, (15, 15), "constant", None)
        x_158 = None
        x_159 = torch.conv1d(
            new_x_4,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_4 = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_161 = torch.nn.functional.silu(x_160, inplace=False)
        x_160 = None
        x_162 = torch.conv1d(
            x_161,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_161 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_163 = x_162.transpose(1, 2)
        x_162 = None
        dropout_33 = torch.nn.functional.dropout(x_163, 0.1, False, False)
        x_163 = None
        x_164 = x_153 + dropout_33
        x_153 = dropout_33 = None
        x_165 = torch.nn.functional.layer_norm(
            x_164,
            (196,),
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_164 = (
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_
        ) = None
        scale_19 = l_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_19 = l_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_4_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_29 = x_165 * scale_19
        scale_19 = None
        x_166 = mul_29 + bias_19
        mul_29 = bias_19 = None
        x_167 = torch._C._nn.linear(
            x_166,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_166 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_168 = torch.nn.functional.silu(x_167, inplace=False)
        x_167 = None
        x_169 = torch.nn.functional.dropout(x_168, 0.1, False, False)
        x_168 = None
        x_170 = torch._C._nn.linear(
            x_169,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_169 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_35 = torch.nn.functional.dropout(x_170, 0.1, False, False)
        x_170 = None
        mul_30 = dropout_35 * 1.0
        dropout_35 = None
        x_171 = x_165 + mul_30
        x_165 = mul_30 = None
        x_172 = torch.nn.functional.layer_norm(
            x_171,
            (196,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_171 = l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_20 = l_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_20 = l_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_31 = x_172 * scale_20
        scale_20 = None
        x_173 = mul_31 + bias_20
        mul_31 = bias_20 = None
        linear_46 = torch._C._nn.linear(
            x_173,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_15 = linear_46.view(1, -1, 4, 49)
        linear_46 = None
        linear_47 = torch._C._nn.linear(
            x_173,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_10 = linear_47.view(1, -1, 4, 49)
        linear_47 = None
        linear_48 = torch._C._nn.linear(
            x_173,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_173 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_10 = linear_48.view(1, -1, 4, 49)
        linear_48 = None
        q_16 = q_15.transpose(1, 2)
        q_15 = None
        k_11 = k_10.transpose(1, 2)
        k_10 = None
        v_11 = v_10.transpose(1, 2)
        v_10 = None
        q_17 = q_16.transpose(1, 2)
        q_16 = None
        linear_49 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_10 = linear_49.view(1, -1, 4, 49)
        linear_49 = None
        p_11 = p_10.transpose(1, 2)
        p_10 = None
        add_60 = (
            q_17
            + l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_5 = add_60.transpose(1, 2)
        add_60 = None
        add_61 = (
            q_17
            + l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_
        )
        q_17 = (
            l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_5 = add_61.transpose(1, 2)
        add_61 = None
        transpose_70 = p_11.transpose(-2, -1)
        p_11 = None
        matrix_bd_10 = torch.matmul(q_with_bias_v_5, transpose_70)
        q_with_bias_v_5 = transpose_70 = None
        x_174 = torch._C._nn.pad(matrix_bd_10, (1, 0), "constant", None)
        matrix_bd_10 = None
        x_175 = x_174.view(1, 4, -1, 131)
        x_174 = None
        getitem_12 = x_175[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_175 = None
        x_176 = getitem_12.view(1, 4, 131, 261)
        getitem_12 = None
        transpose_71 = k_11.transpose(-2, -1)
        k_11 = None
        matrix_ac_5 = torch.matmul(q_with_bias_u_5, transpose_71)
        q_with_bias_u_5 = transpose_71 = None
        matrix_bd_11 = x_176[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_176 = None
        add_62 = matrix_ac_5 + matrix_bd_11
        matrix_ac_5 = matrix_bd_11 = None
        scores_10 = add_62 / 7.0
        add_62 = None
        mask_6 = att_mask_2.unsqueeze(1)
        scores_11 = scores_10.masked_fill(mask_6, -10000.0)
        scores_10 = None
        softmax_5 = torch.softmax(scores_11, dim=-1)
        scores_11 = None
        attn_5 = softmax_5.masked_fill(mask_6, 0.0)
        softmax_5 = mask_6 = None
        p_attn_5 = torch.nn.functional.dropout(attn_5, 0.1, False, False)
        attn_5 = None
        x_177 = torch.matmul(p_attn_5, v_11)
        p_attn_5 = v_11 = None
        transpose_72 = x_177.transpose(1, 2)
        x_177 = None
        x_178 = transpose_72.reshape(1, -1, 196)
        transpose_72 = None
        out_5 = torch._C._nn.linear(
            x_178,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_178 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_37 = torch.nn.functional.dropout(out_5, 0.1, False, False)
        out_5 = None
        x_179 = x_172 + dropout_37
        x_172 = dropout_37 = None
        x_180 = torch.nn.functional.layer_norm(
            x_179,
            (196,),
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_179 = (
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_
        ) = None
        scale_21 = l_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_21 = l_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_32 = x_180 * scale_21
        scale_21 = None
        x_181 = mul_32 + bias_21
        mul_32 = bias_21 = None
        x_182 = torch._C._nn.linear(
            x_181,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_181 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_183 = torch.nn.functional.silu(x_182, inplace=False)
        x_182 = None
        x_184 = torch.nn.functional.dropout(x_183, 0.1, False, False)
        x_183 = None
        x_185 = torch._C._nn.linear(
            x_184,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_184 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_39 = torch.nn.functional.dropout(x_185, 0.1, False, False)
        x_185 = None
        mul_33 = dropout_39 * 1.0
        dropout_39 = None
        x_186 = x_180 + mul_33
        x_180 = mul_33 = None
        x_187 = torch.nn.functional.layer_norm(
            x_186,
            (196,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_186 = l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_22 = l_instance_modules_layers_modules_5_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_conv_scale_parameters_scale_ = None
        bias_22 = l_instance_modules_layers_modules_5_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_conv_scale_parameters_bias_ = None
        mul_34 = x_187 * scale_22
        scale_22 = None
        x_188 = mul_34 + bias_22
        mul_34 = bias_22 = None
        x_189 = x_188.transpose(1, 2)
        x_188 = None
        x_190 = torch.conv1d(
            x_189,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_189 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_191 = torch.nn.functional.silu(x_190, inplace=True)
        x_190 = None
        unsqueeze_14 = pad_mask.unsqueeze(1)
        x_192 = x_191.masked_fill(unsqueeze_14, 0.0)
        x_191 = unsqueeze_14 = None
        new_x_5 = torch._C._nn.pad(x_192, (15, 15), "constant", None)
        x_192 = None
        x_193 = torch.conv1d(
            new_x_5,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_5 = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_195 = torch.nn.functional.silu(x_194, inplace=False)
        x_194 = None
        x_196 = torch.conv1d(
            x_195,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_195 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_197 = x_196.transpose(1, 2)
        x_196 = None
        dropout_40 = torch.nn.functional.dropout(x_197, 0.1, False, False)
        x_197 = None
        x_198 = x_187 + dropout_40
        x_187 = dropout_40 = None
        x_199 = torch.nn.functional.layer_norm(
            x_198,
            (196,),
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_198 = (
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_
        ) = None
        scale_23 = l_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_23 = l_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_5_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_35 = x_199 * scale_23
        scale_23 = None
        x_200 = mul_35 + bias_23
        mul_35 = bias_23 = None
        x_201 = torch._C._nn.linear(
            x_200,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_200 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_202 = torch.nn.functional.silu(x_201, inplace=False)
        x_201 = None
        x_203 = torch.nn.functional.dropout(x_202, 0.1, False, False)
        x_202 = None
        x_204 = torch._C._nn.linear(
            x_203,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_203 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_42 = torch.nn.functional.dropout(x_204, 0.1, False, False)
        x_204 = None
        mul_36 = dropout_42 * 1.0
        dropout_42 = None
        x_205 = x_199 + mul_36
        x_199 = mul_36 = None
        x_206 = torch.nn.functional.layer_norm(
            x_205,
            (196,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_205 = l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_24 = l_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_24 = l_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_37 = x_206 * scale_24
        scale_24 = None
        x_207 = mul_37 + bias_24
        mul_37 = bias_24 = None
        linear_55 = torch._C._nn.linear(
            x_207,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_18 = linear_55.view(1, -1, 4, 49)
        linear_55 = None
        linear_56 = torch._C._nn.linear(
            x_207,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_12 = linear_56.view(1, -1, 4, 49)
        linear_56 = None
        linear_57 = torch._C._nn.linear(
            x_207,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_207 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_12 = linear_57.view(1, -1, 4, 49)
        linear_57 = None
        q_19 = q_18.transpose(1, 2)
        q_18 = None
        k_13 = k_12.transpose(1, 2)
        k_12 = None
        v_13 = v_12.transpose(1, 2)
        v_12 = None
        q_20 = q_19.transpose(1, 2)
        q_19 = None
        linear_58 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_12 = linear_58.view(1, -1, 4, 49)
        linear_58 = None
        p_13 = p_12.transpose(1, 2)
        p_12 = None
        add_71 = (
            q_20
            + l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_6 = add_71.transpose(1, 2)
        add_71 = None
        add_72 = (
            q_20
            + l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_
        )
        q_20 = (
            l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_6 = add_72.transpose(1, 2)
        add_72 = None
        transpose_82 = p_13.transpose(-2, -1)
        p_13 = None
        matrix_bd_12 = torch.matmul(q_with_bias_v_6, transpose_82)
        q_with_bias_v_6 = transpose_82 = None
        x_208 = torch._C._nn.pad(matrix_bd_12, (1, 0), "constant", None)
        matrix_bd_12 = None
        x_209 = x_208.view(1, 4, -1, 131)
        x_208 = None
        getitem_14 = x_209[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_209 = None
        x_210 = getitem_14.view(1, 4, 131, 261)
        getitem_14 = None
        transpose_83 = k_13.transpose(-2, -1)
        k_13 = None
        matrix_ac_6 = torch.matmul(q_with_bias_u_6, transpose_83)
        q_with_bias_u_6 = transpose_83 = None
        matrix_bd_13 = x_210[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_210 = None
        add_73 = matrix_ac_6 + matrix_bd_13
        matrix_ac_6 = matrix_bd_13 = None
        scores_12 = add_73 / 7.0
        add_73 = None
        mask_7 = att_mask_2.unsqueeze(1)
        scores_13 = scores_12.masked_fill(mask_7, -10000.0)
        scores_12 = None
        softmax_6 = torch.softmax(scores_13, dim=-1)
        scores_13 = None
        attn_6 = softmax_6.masked_fill(mask_7, 0.0)
        softmax_6 = mask_7 = None
        p_attn_6 = torch.nn.functional.dropout(attn_6, 0.1, False, False)
        attn_6 = None
        x_211 = torch.matmul(p_attn_6, v_13)
        p_attn_6 = v_13 = None
        transpose_84 = x_211.transpose(1, 2)
        x_211 = None
        x_212 = transpose_84.reshape(1, -1, 196)
        transpose_84 = None
        out_6 = torch._C._nn.linear(
            x_212,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_212 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_44 = torch.nn.functional.dropout(out_6, 0.1, False, False)
        out_6 = None
        x_213 = x_206 + dropout_44
        x_206 = dropout_44 = None
        x_214 = torch.nn.functional.layer_norm(
            x_213,
            (196,),
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_213 = (
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_
        ) = None
        scale_25 = l_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_25 = l_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_38 = x_214 * scale_25
        scale_25 = None
        x_215 = mul_38 + bias_25
        mul_38 = bias_25 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_215 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_217 = torch.nn.functional.silu(x_216, inplace=False)
        x_216 = None
        x_218 = torch.nn.functional.dropout(x_217, 0.1, False, False)
        x_217 = None
        x_219 = torch._C._nn.linear(
            x_218,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_218 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_46 = torch.nn.functional.dropout(x_219, 0.1, False, False)
        x_219 = None
        mul_39 = dropout_46 * 1.0
        dropout_46 = None
        x_220 = x_214 + mul_39
        x_214 = mul_39 = None
        x_221 = torch.nn.functional.layer_norm(
            x_220,
            (196,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_220 = l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_26 = l_instance_modules_layers_modules_6_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_conv_scale_parameters_scale_ = None
        bias_26 = l_instance_modules_layers_modules_6_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_conv_scale_parameters_bias_ = None
        mul_40 = x_221 * scale_26
        scale_26 = None
        x_222 = mul_40 + bias_26
        mul_40 = bias_26 = None
        x_223 = x_222.transpose(1, 2)
        x_222 = None
        x_224 = torch.conv1d(
            x_223,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_223 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_225 = torch.nn.functional.silu(x_224, inplace=True)
        x_224 = None
        unsqueeze_16 = pad_mask.unsqueeze(1)
        x_226 = x_225.masked_fill(unsqueeze_16, 0.0)
        x_225 = unsqueeze_16 = None
        new_x_6 = torch._C._nn.pad(x_226, (15, 15), "constant", None)
        x_226 = None
        x_227 = torch.conv1d(
            new_x_6,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_6 = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_229 = torch.nn.functional.silu(x_228, inplace=False)
        x_228 = None
        x_230 = torch.conv1d(
            x_229,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_229 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_231 = x_230.transpose(1, 2)
        x_230 = None
        dropout_47 = torch.nn.functional.dropout(x_231, 0.1, False, False)
        x_231 = None
        x_232 = x_221 + dropout_47
        x_221 = dropout_47 = None
        x_233 = torch.nn.functional.layer_norm(
            x_232,
            (196,),
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_232 = (
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_
        ) = None
        scale_27 = l_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_27 = l_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_6_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_41 = x_233 * scale_27
        scale_27 = None
        x_234 = mul_41 + bias_27
        mul_41 = bias_27 = None
        x_235 = torch._C._nn.linear(
            x_234,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_234 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_236 = torch.nn.functional.silu(x_235, inplace=False)
        x_235 = None
        x_237 = torch.nn.functional.dropout(x_236, 0.1, False, False)
        x_236 = None
        x_238 = torch._C._nn.linear(
            x_237,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_237 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_49 = torch.nn.functional.dropout(x_238, 0.1, False, False)
        x_238 = None
        mul_42 = dropout_49 * 1.0
        dropout_49 = None
        x_239 = x_233 + mul_42
        x_233 = mul_42 = None
        x_240 = torch.nn.functional.layer_norm(
            x_239,
            (196,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_239 = l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_28 = l_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_28 = l_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_43 = x_240 * scale_28
        scale_28 = None
        x_241 = mul_43 + bias_28
        mul_43 = bias_28 = None
        linear_64 = torch._C._nn.linear(
            x_241,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_21 = linear_64.view(1, -1, 4, 49)
        linear_64 = None
        linear_65 = torch._C._nn.linear(
            x_241,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_14 = linear_65.view(1, -1, 4, 49)
        linear_65 = None
        linear_66 = torch._C._nn.linear(
            x_241,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_241 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_14 = linear_66.view(1, -1, 4, 49)
        linear_66 = None
        q_22 = q_21.transpose(1, 2)
        q_21 = None
        k_15 = k_14.transpose(1, 2)
        k_14 = None
        v_15 = v_14.transpose(1, 2)
        v_14 = None
        q_23 = q_22.transpose(1, 2)
        q_22 = None
        linear_67 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_14 = linear_67.view(1, -1, 4, 49)
        linear_67 = None
        p_15 = p_14.transpose(1, 2)
        p_14 = None
        add_82 = (
            q_23
            + l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_7 = add_82.transpose(1, 2)
        add_82 = None
        add_83 = (
            q_23
            + l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_
        )
        q_23 = (
            l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_7 = add_83.transpose(1, 2)
        add_83 = None
        transpose_94 = p_15.transpose(-2, -1)
        p_15 = None
        matrix_bd_14 = torch.matmul(q_with_bias_v_7, transpose_94)
        q_with_bias_v_7 = transpose_94 = None
        x_242 = torch._C._nn.pad(matrix_bd_14, (1, 0), "constant", None)
        matrix_bd_14 = None
        x_243 = x_242.view(1, 4, -1, 131)
        x_242 = None
        getitem_16 = x_243[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_243 = None
        x_244 = getitem_16.view(1, 4, 131, 261)
        getitem_16 = None
        transpose_95 = k_15.transpose(-2, -1)
        k_15 = None
        matrix_ac_7 = torch.matmul(q_with_bias_u_7, transpose_95)
        q_with_bias_u_7 = transpose_95 = None
        matrix_bd_15 = x_244[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_244 = None
        add_84 = matrix_ac_7 + matrix_bd_15
        matrix_ac_7 = matrix_bd_15 = None
        scores_14 = add_84 / 7.0
        add_84 = None
        mask_8 = att_mask_2.unsqueeze(1)
        scores_15 = scores_14.masked_fill(mask_8, -10000.0)
        scores_14 = None
        softmax_7 = torch.softmax(scores_15, dim=-1)
        scores_15 = None
        attn_7 = softmax_7.masked_fill(mask_8, 0.0)
        softmax_7 = mask_8 = None
        p_attn_7 = torch.nn.functional.dropout(attn_7, 0.1, False, False)
        attn_7 = None
        x_245 = torch.matmul(p_attn_7, v_15)
        p_attn_7 = v_15 = None
        transpose_96 = x_245.transpose(1, 2)
        x_245 = None
        x_246 = transpose_96.reshape(1, -1, 196)
        transpose_96 = None
        out_7 = torch._C._nn.linear(
            x_246,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_246 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_51 = torch.nn.functional.dropout(out_7, 0.1, False, False)
        out_7 = None
        x_247 = x_240 + dropout_51
        x_240 = dropout_51 = None
        x_248 = torch.nn.functional.layer_norm(
            x_247,
            (196,),
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_247 = (
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_
        ) = None
        scale_29 = l_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_29 = l_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_44 = x_248 * scale_29
        scale_29 = None
        x_249 = mul_44 + bias_29
        mul_44 = bias_29 = None
        x_250 = torch._C._nn.linear(
            x_249,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_249 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_251 = torch.nn.functional.silu(x_250, inplace=False)
        x_250 = None
        x_252 = torch.nn.functional.dropout(x_251, 0.1, False, False)
        x_251 = None
        x_253 = torch._C._nn.linear(
            x_252,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_252 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_53 = torch.nn.functional.dropout(x_253, 0.1, False, False)
        x_253 = None
        mul_45 = dropout_53 * 1.0
        dropout_53 = None
        x_254 = x_248 + mul_45
        x_248 = mul_45 = None
        x_255 = torch.nn.functional.layer_norm(
            x_254,
            (196,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_254 = l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_30 = l_instance_modules_layers_modules_7_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_conv_scale_parameters_scale_ = None
        bias_30 = l_instance_modules_layers_modules_7_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_conv_scale_parameters_bias_ = None
        mul_46 = x_255 * scale_30
        scale_30 = None
        x_256 = mul_46 + bias_30
        mul_46 = bias_30 = None
        x_257 = x_256.transpose(1, 2)
        x_256 = None
        x_258 = torch.conv1d(
            x_257,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_257 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_259 = torch.nn.functional.silu(x_258, inplace=True)
        x_258 = None
        unsqueeze_18 = pad_mask.unsqueeze(1)
        x_260 = x_259.masked_fill(unsqueeze_18, 0.0)
        x_259 = unsqueeze_18 = None
        new_x_7 = torch._C._nn.pad(x_260, (15, 15), "constant", None)
        x_260 = None
        x_261 = torch.conv1d(
            new_x_7,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_7 = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_261 = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_263 = torch.nn.functional.silu(x_262, inplace=False)
        x_262 = None
        x_264 = torch.conv1d(
            x_263,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_263 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_265 = x_264.transpose(1, 2)
        x_264 = None
        dropout_54 = torch.nn.functional.dropout(x_265, 0.1, False, False)
        x_265 = None
        x_266 = x_255 + dropout_54
        x_255 = dropout_54 = None
        x_267 = torch.nn.functional.layer_norm(
            x_266,
            (196,),
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_266 = (
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_
        ) = None
        scale_31 = l_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_31 = l_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_7_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_47 = x_267 * scale_31
        scale_31 = None
        x_268 = mul_47 + bias_31
        mul_47 = bias_31 = None
        x_269 = torch._C._nn.linear(
            x_268,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_268 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_270 = torch.nn.functional.silu(x_269, inplace=False)
        x_269 = None
        x_271 = torch.nn.functional.dropout(x_270, 0.1, False, False)
        x_270 = None
        x_272 = torch._C._nn.linear(
            x_271,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_271 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_56 = torch.nn.functional.dropout(x_272, 0.1, False, False)
        x_272 = None
        mul_48 = dropout_56 * 1.0
        dropout_56 = None
        x_273 = x_267 + mul_48
        x_267 = mul_48 = None
        x_274 = torch.nn.functional.layer_norm(
            x_273,
            (196,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_273 = l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_275 = x_274.transpose(1, 2)
        float_1 = x_275.float()
        x_275 = None
        unsqueeze_19 = pad_mask.unsqueeze(1)
        x_276 = float_1.masked_fill(unsqueeze_19, 0.0)
        float_1 = unsqueeze_19 = None
        x_277 = torch.conv1d(
            x_276,
            l_instance_modules_time_reduce_layer_modules_dw_conv_parameters_weight_,
            l_instance_modules_time_reduce_layer_modules_dw_conv_parameters_bias_,
            (2,),
            (3,),
            (1,),
            196,
        )
        x_276 = (
            l_instance_modules_time_reduce_layer_modules_dw_conv_parameters_weight_
        ) = l_instance_modules_time_reduce_layer_modules_dw_conv_parameters_bias_ = None
        x_278 = torch.conv1d(
            x_277,
            l_instance_modules_time_reduce_layer_modules_pw_conv_parameters_weight_,
            l_instance_modules_time_reduce_layer_modules_pw_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_277 = (
            l_instance_modules_time_reduce_layer_modules_pw_conv_parameters_weight_
        ) = l_instance_modules_time_reduce_layer_modules_pw_conv_parameters_bias_ = None
        x_279 = x_278.transpose(1, 2)
        x_278 = None
        att_mask_3 = att_mask_2[
            (slice(None, None, None), slice(None, None, 2), slice(None, None, 2))
        ]
        pad_mask_1 = pad_mask[(slice(None, None, None), slice(None, None, 2))]
        x_280 = torch._C._nn.pad(x_279, (0, 0, 0, -1), "constant", None)
        x_279 = None
        pos_emb_1 = l_instance_modules_time_reduce_pos_enc_buffers_pe_[
            (slice(None, None, None), slice(4934, 5065, None))
        ]
        l_instance_modules_time_reduce_pos_enc_buffers_pe_ = None
        _ = torch.nn.functional.dropout(x_280, 0.0, False, False)
        _ = None
        scale_32 = l_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_32 = l_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_49 = x_280 * scale_32
        scale_32 = None
        x_281 = mul_49 + bias_32
        mul_49 = bias_32 = None
        linear_73 = torch._C._nn.linear(
            x_281,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_24 = linear_73.view(1, -1, 4, 49)
        linear_73 = None
        linear_74 = torch._C._nn.linear(
            x_281,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_16 = linear_74.view(1, -1, 4, 49)
        linear_74 = None
        linear_75 = torch._C._nn.linear(
            x_281,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_281 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_16 = linear_75.view(1, -1, 4, 49)
        linear_75 = None
        q_25 = q_24.transpose(1, 2)
        q_24 = None
        k_17 = k_16.transpose(1, 2)
        k_16 = None
        v_17 = v_16.transpose(1, 2)
        v_16 = None
        q_26 = q_25.transpose(1, 2)
        q_25 = None
        linear_76 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_16 = linear_76.view(1, -1, 4, 49)
        linear_76 = None
        p_17 = p_16.transpose(1, 2)
        p_16 = None
        add_93 = (
            q_26
            + l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_8 = add_93.transpose(1, 2)
        add_93 = None
        add_94 = (
            q_26
            + l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_
        )
        q_26 = (
            l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_8 = add_94.transpose(1, 2)
        add_94 = None
        transpose_108 = p_17.transpose(-2, -1)
        p_17 = None
        matrix_bd_16 = torch.matmul(q_with_bias_v_8, transpose_108)
        q_with_bias_v_8 = transpose_108 = None
        x_282 = torch._C._nn.pad(matrix_bd_16, (1, 0), "constant", None)
        matrix_bd_16 = None
        x_283 = x_282.view(1, 4, -1, 66)
        x_282 = None
        getitem_21 = x_283[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_283 = None
        x_284 = getitem_21.view(1, 4, 66, 131)
        getitem_21 = None
        transpose_109 = k_17.transpose(-2, -1)
        k_17 = None
        matrix_ac_8 = torch.matmul(q_with_bias_u_8, transpose_109)
        q_with_bias_u_8 = transpose_109 = None
        matrix_bd_17 = x_284[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_284 = None
        add_95 = matrix_ac_8 + matrix_bd_17
        matrix_ac_8 = matrix_bd_17 = None
        scores_16 = add_95 / 7.0
        add_95 = None
        mask_9 = att_mask_3.unsqueeze(1)
        scores_17 = scores_16.masked_fill(mask_9, -10000.0)
        scores_16 = None
        softmax_8 = torch.softmax(scores_17, dim=-1)
        scores_17 = None
        attn_8 = softmax_8.masked_fill(mask_9, 0.0)
        softmax_8 = mask_9 = None
        p_attn_8 = torch.nn.functional.dropout(attn_8, 0.1, False, False)
        attn_8 = None
        x_285 = torch.matmul(p_attn_8, v_17)
        p_attn_8 = v_17 = None
        transpose_110 = x_285.transpose(1, 2)
        x_285 = None
        x_286 = transpose_110.reshape(1, -1, 196)
        transpose_110 = None
        out_8 = torch._C._nn.linear(
            x_286,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_286 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_59 = torch.nn.functional.dropout(out_8, 0.1, False, False)
        out_8 = None
        x_287 = x_280 + dropout_59
        x_280 = dropout_59 = None
        x_288 = torch.nn.functional.layer_norm(
            x_287,
            (196,),
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_287 = (
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_
        ) = None
        scale_33 = l_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_33 = l_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_50 = x_288 * scale_33
        scale_33 = None
        x_289 = mul_50 + bias_33
        mul_50 = bias_33 = None
        x_290 = torch._C._nn.linear(
            x_289,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_289 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_291 = torch.nn.functional.silu(x_290, inplace=False)
        x_290 = None
        x_292 = torch.nn.functional.dropout(x_291, 0.1, False, False)
        x_291 = None
        x_293 = torch._C._nn.linear(
            x_292,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_292 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_61 = torch.nn.functional.dropout(x_293, 0.1, False, False)
        x_293 = None
        mul_51 = dropout_61 * 1.0
        dropout_61 = None
        x_294 = x_288 + mul_51
        x_288 = mul_51 = None
        x_295 = torch.nn.functional.layer_norm(
            x_294,
            (196,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_294 = l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_34 = l_instance_modules_layers_modules_8_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_conv_scale_parameters_scale_ = None
        bias_34 = l_instance_modules_layers_modules_8_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_conv_scale_parameters_bias_ = None
        mul_52 = x_295 * scale_34
        scale_34 = None
        x_296 = mul_52 + bias_34
        mul_52 = bias_34 = None
        x_297 = x_296.transpose(1, 2)
        x_296 = None
        x_298 = torch.conv1d(
            x_297,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_297 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_299 = torch.nn.functional.silu(x_298, inplace=True)
        x_298 = None
        unsqueeze_21 = pad_mask_1.unsqueeze(1)
        x_300 = x_299.masked_fill(unsqueeze_21, 0.0)
        x_299 = unsqueeze_21 = None
        new_x_8 = torch._C._nn.pad(x_300, (15, 15), "constant", None)
        x_300 = None
        x_301 = torch.conv1d(
            new_x_8,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_8 = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_303 = torch.nn.functional.silu(x_302, inplace=False)
        x_302 = None
        x_304 = torch.conv1d(
            x_303,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_303 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_305 = x_304.transpose(1, 2)
        x_304 = None
        dropout_62 = torch.nn.functional.dropout(x_305, 0.1, False, False)
        x_305 = None
        x_306 = x_295 + dropout_62
        x_295 = dropout_62 = None
        x_307 = torch.nn.functional.layer_norm(
            x_306,
            (196,),
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_306 = (
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_
        ) = None
        scale_35 = l_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_35 = l_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_8_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_53 = x_307 * scale_35
        scale_35 = None
        x_308 = mul_53 + bias_35
        mul_53 = bias_35 = None
        x_309 = torch._C._nn.linear(
            x_308,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_308 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_310 = torch.nn.functional.silu(x_309, inplace=False)
        x_309 = None
        x_311 = torch.nn.functional.dropout(x_310, 0.1, False, False)
        x_310 = None
        x_312 = torch._C._nn.linear(
            x_311,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_311 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_64 = torch.nn.functional.dropout(x_312, 0.1, False, False)
        x_312 = None
        mul_54 = dropout_64 * 1.0
        dropout_64 = None
        x_313 = x_307 + mul_54
        x_307 = mul_54 = None
        x_314 = torch.nn.functional.layer_norm(
            x_313,
            (196,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_313 = l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_36 = l_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_36 = l_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_55 = x_314 * scale_36
        scale_36 = None
        x_315 = mul_55 + bias_36
        mul_55 = bias_36 = None
        linear_82 = torch._C._nn.linear(
            x_315,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_27 = linear_82.view(1, -1, 4, 49)
        linear_82 = None
        linear_83 = torch._C._nn.linear(
            x_315,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_18 = linear_83.view(1, -1, 4, 49)
        linear_83 = None
        linear_84 = torch._C._nn.linear(
            x_315,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_315 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_18 = linear_84.view(1, -1, 4, 49)
        linear_84 = None
        q_28 = q_27.transpose(1, 2)
        q_27 = None
        k_19 = k_18.transpose(1, 2)
        k_18 = None
        v_19 = v_18.transpose(1, 2)
        v_18 = None
        q_29 = q_28.transpose(1, 2)
        q_28 = None
        linear_85 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_18 = linear_85.view(1, -1, 4, 49)
        linear_85 = None
        p_19 = p_18.transpose(1, 2)
        p_18 = None
        add_104 = (
            q_29
            + l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_9 = add_104.transpose(1, 2)
        add_104 = None
        add_105 = (
            q_29
            + l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_
        )
        q_29 = (
            l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_9 = add_105.transpose(1, 2)
        add_105 = None
        transpose_120 = p_19.transpose(-2, -1)
        p_19 = None
        matrix_bd_18 = torch.matmul(q_with_bias_v_9, transpose_120)
        q_with_bias_v_9 = transpose_120 = None
        x_316 = torch._C._nn.pad(matrix_bd_18, (1, 0), "constant", None)
        matrix_bd_18 = None
        x_317 = x_316.view(1, 4, -1, 66)
        x_316 = None
        getitem_23 = x_317[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_317 = None
        x_318 = getitem_23.view(1, 4, 66, 131)
        getitem_23 = None
        transpose_121 = k_19.transpose(-2, -1)
        k_19 = None
        matrix_ac_9 = torch.matmul(q_with_bias_u_9, transpose_121)
        q_with_bias_u_9 = transpose_121 = None
        matrix_bd_19 = x_318[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_318 = None
        add_106 = matrix_ac_9 + matrix_bd_19
        matrix_ac_9 = matrix_bd_19 = None
        scores_18 = add_106 / 7.0
        add_106 = None
        mask_10 = att_mask_3.unsqueeze(1)
        scores_19 = scores_18.masked_fill(mask_10, -10000.0)
        scores_18 = None
        softmax_9 = torch.softmax(scores_19, dim=-1)
        scores_19 = None
        attn_9 = softmax_9.masked_fill(mask_10, 0.0)
        softmax_9 = mask_10 = None
        p_attn_9 = torch.nn.functional.dropout(attn_9, 0.1, False, False)
        attn_9 = None
        x_319 = torch.matmul(p_attn_9, v_19)
        p_attn_9 = v_19 = None
        transpose_122 = x_319.transpose(1, 2)
        x_319 = None
        x_320 = transpose_122.reshape(1, -1, 196)
        transpose_122 = None
        out_9 = torch._C._nn.linear(
            x_320,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_320 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_66 = torch.nn.functional.dropout(out_9, 0.1, False, False)
        out_9 = None
        x_321 = x_314 + dropout_66
        x_314 = dropout_66 = None
        x_322 = torch.nn.functional.layer_norm(
            x_321,
            (196,),
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_321 = (
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_
        ) = None
        scale_37 = l_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_37 = l_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_56 = x_322 * scale_37
        scale_37 = None
        x_323 = mul_56 + bias_37
        mul_56 = bias_37 = None
        x_324 = torch._C._nn.linear(
            x_323,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_323 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_325 = torch.nn.functional.silu(x_324, inplace=False)
        x_324 = None
        x_326 = torch.nn.functional.dropout(x_325, 0.1, False, False)
        x_325 = None
        x_327 = torch._C._nn.linear(
            x_326,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_326 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_68 = torch.nn.functional.dropout(x_327, 0.1, False, False)
        x_327 = None
        mul_57 = dropout_68 * 1.0
        dropout_68 = None
        x_328 = x_322 + mul_57
        x_322 = mul_57 = None
        x_329 = torch.nn.functional.layer_norm(
            x_328,
            (196,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_328 = l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_38 = l_instance_modules_layers_modules_9_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_conv_scale_parameters_scale_ = None
        bias_38 = l_instance_modules_layers_modules_9_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_conv_scale_parameters_bias_ = None
        mul_58 = x_329 * scale_38
        scale_38 = None
        x_330 = mul_58 + bias_38
        mul_58 = bias_38 = None
        x_331 = x_330.transpose(1, 2)
        x_330 = None
        x_332 = torch.conv1d(
            x_331,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_331 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_333 = torch.nn.functional.silu(x_332, inplace=True)
        x_332 = None
        unsqueeze_23 = pad_mask_1.unsqueeze(1)
        x_334 = x_333.masked_fill(unsqueeze_23, 0.0)
        x_333 = unsqueeze_23 = None
        new_x_9 = torch._C._nn.pad(x_334, (15, 15), "constant", None)
        x_334 = None
        x_335 = torch.conv1d(
            new_x_9,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_9 = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_336 = torch.nn.functional.batch_norm(
            x_335,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_335 = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_337 = torch.nn.functional.silu(x_336, inplace=False)
        x_336 = None
        x_338 = torch.conv1d(
            x_337,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_337 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_339 = x_338.transpose(1, 2)
        x_338 = None
        dropout_69 = torch.nn.functional.dropout(x_339, 0.1, False, False)
        x_339 = None
        x_340 = x_329 + dropout_69
        x_329 = dropout_69 = None
        x_341 = torch.nn.functional.layer_norm(
            x_340,
            (196,),
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_340 = (
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_
        ) = None
        scale_39 = l_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_39 = l_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_9_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_59 = x_341 * scale_39
        scale_39 = None
        x_342 = mul_59 + bias_39
        mul_59 = bias_39 = None
        x_343 = torch._C._nn.linear(
            x_342,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_342 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_344 = torch.nn.functional.silu(x_343, inplace=False)
        x_343 = None
        x_345 = torch.nn.functional.dropout(x_344, 0.1, False, False)
        x_344 = None
        x_346 = torch._C._nn.linear(
            x_345,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_345 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_71 = torch.nn.functional.dropout(x_346, 0.1, False, False)
        x_346 = None
        mul_60 = dropout_71 * 1.0
        dropout_71 = None
        x_347 = x_341 + mul_60
        x_341 = mul_60 = None
        x_348 = torch.nn.functional.layer_norm(
            x_347,
            (196,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_347 = l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_40 = l_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_40 = l_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_61 = x_348 * scale_40
        scale_40 = None
        x_349 = mul_61 + bias_40
        mul_61 = bias_40 = None
        linear_91 = torch._C._nn.linear(
            x_349,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_30 = linear_91.view(1, -1, 4, 49)
        linear_91 = None
        linear_92 = torch._C._nn.linear(
            x_349,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_20 = linear_92.view(1, -1, 4, 49)
        linear_92 = None
        linear_93 = torch._C._nn.linear(
            x_349,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_349 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_20 = linear_93.view(1, -1, 4, 49)
        linear_93 = None
        q_31 = q_30.transpose(1, 2)
        q_30 = None
        k_21 = k_20.transpose(1, 2)
        k_20 = None
        v_21 = v_20.transpose(1, 2)
        v_20 = None
        q_32 = q_31.transpose(1, 2)
        q_31 = None
        linear_94 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_20 = linear_94.view(1, -1, 4, 49)
        linear_94 = None
        p_21 = p_20.transpose(1, 2)
        p_20 = None
        add_115 = (
            q_32
            + l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_10 = add_115.transpose(1, 2)
        add_115 = None
        add_116 = (
            q_32
            + l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_
        )
        q_32 = l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_10 = add_116.transpose(1, 2)
        add_116 = None
        transpose_132 = p_21.transpose(-2, -1)
        p_21 = None
        matrix_bd_20 = torch.matmul(q_with_bias_v_10, transpose_132)
        q_with_bias_v_10 = transpose_132 = None
        x_350 = torch._C._nn.pad(matrix_bd_20, (1, 0), "constant", None)
        matrix_bd_20 = None
        x_351 = x_350.view(1, 4, -1, 66)
        x_350 = None
        getitem_25 = x_351[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_351 = None
        x_352 = getitem_25.view(1, 4, 66, 131)
        getitem_25 = None
        transpose_133 = k_21.transpose(-2, -1)
        k_21 = None
        matrix_ac_10 = torch.matmul(q_with_bias_u_10, transpose_133)
        q_with_bias_u_10 = transpose_133 = None
        matrix_bd_21 = x_352[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_352 = None
        add_117 = matrix_ac_10 + matrix_bd_21
        matrix_ac_10 = matrix_bd_21 = None
        scores_20 = add_117 / 7.0
        add_117 = None
        mask_11 = att_mask_3.unsqueeze(1)
        scores_21 = scores_20.masked_fill(mask_11, -10000.0)
        scores_20 = None
        softmax_10 = torch.softmax(scores_21, dim=-1)
        scores_21 = None
        attn_10 = softmax_10.masked_fill(mask_11, 0.0)
        softmax_10 = mask_11 = None
        p_attn_10 = torch.nn.functional.dropout(attn_10, 0.1, False, False)
        attn_10 = None
        x_353 = torch.matmul(p_attn_10, v_21)
        p_attn_10 = v_21 = None
        transpose_134 = x_353.transpose(1, 2)
        x_353 = None
        x_354 = transpose_134.reshape(1, -1, 196)
        transpose_134 = None
        out_10 = torch._C._nn.linear(
            x_354,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_354 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_73 = torch.nn.functional.dropout(out_10, 0.1, False, False)
        out_10 = None
        x_355 = x_348 + dropout_73
        x_348 = dropout_73 = None
        x_356 = torch.nn.functional.layer_norm(
            x_355,
            (196,),
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_355 = l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_
        ) = None
        scale_41 = l_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_41 = l_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_62 = x_356 * scale_41
        scale_41 = None
        x_357 = mul_62 + bias_41
        mul_62 = bias_41 = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_357 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_359 = torch.nn.functional.silu(x_358, inplace=False)
        x_358 = None
        x_360 = torch.nn.functional.dropout(x_359, 0.1, False, False)
        x_359 = None
        x_361 = torch._C._nn.linear(
            x_360,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_360 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_75 = torch.nn.functional.dropout(x_361, 0.1, False, False)
        x_361 = None
        mul_63 = dropout_75 * 1.0
        dropout_75 = None
        x_362 = x_356 + mul_63
        x_356 = mul_63 = None
        x_363 = torch.nn.functional.layer_norm(
            x_362,
            (196,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_362 = l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_42 = l_instance_modules_layers_modules_10_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_conv_scale_parameters_scale_ = None
        bias_42 = l_instance_modules_layers_modules_10_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_conv_scale_parameters_bias_ = None
        mul_64 = x_363 * scale_42
        scale_42 = None
        x_364 = mul_64 + bias_42
        mul_64 = bias_42 = None
        x_365 = x_364.transpose(1, 2)
        x_364 = None
        x_366 = torch.conv1d(
            x_365,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_365 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_367 = torch.nn.functional.silu(x_366, inplace=True)
        x_366 = None
        unsqueeze_25 = pad_mask_1.unsqueeze(1)
        x_368 = x_367.masked_fill(unsqueeze_25, 0.0)
        x_367 = unsqueeze_25 = None
        new_x_10 = torch._C._nn.pad(x_368, (15, 15), "constant", None)
        x_368 = None
        x_369 = torch.conv1d(
            new_x_10,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_10 = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_371 = torch.nn.functional.silu(x_370, inplace=False)
        x_370 = None
        x_372 = torch.conv1d(
            x_371,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_371 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_373 = x_372.transpose(1, 2)
        x_372 = None
        dropout_76 = torch.nn.functional.dropout(x_373, 0.1, False, False)
        x_373 = None
        x_374 = x_363 + dropout_76
        x_363 = dropout_76 = None
        x_375 = torch.nn.functional.layer_norm(
            x_374,
            (196,),
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_374 = (
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_
        ) = None
        scale_43 = l_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_43 = l_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_10_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_65 = x_375 * scale_43
        scale_43 = None
        x_376 = mul_65 + bias_43
        mul_65 = bias_43 = None
        x_377 = torch._C._nn.linear(
            x_376,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_376 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_378 = torch.nn.functional.silu(x_377, inplace=False)
        x_377 = None
        x_379 = torch.nn.functional.dropout(x_378, 0.1, False, False)
        x_378 = None
        x_380 = torch._C._nn.linear(
            x_379,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_379 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_78 = torch.nn.functional.dropout(x_380, 0.1, False, False)
        x_380 = None
        mul_66 = dropout_78 * 1.0
        dropout_78 = None
        x_381 = x_375 + mul_66
        x_375 = mul_66 = None
        x_382 = torch.nn.functional.layer_norm(
            x_381,
            (196,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_381 = l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_44 = l_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_44 = l_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_67 = x_382 * scale_44
        scale_44 = None
        x_383 = mul_67 + bias_44
        mul_67 = bias_44 = None
        linear_100 = torch._C._nn.linear(
            x_383,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_33 = linear_100.view(1, -1, 4, 49)
        linear_100 = None
        linear_101 = torch._C._nn.linear(
            x_383,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_22 = linear_101.view(1, -1, 4, 49)
        linear_101 = None
        linear_102 = torch._C._nn.linear(
            x_383,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_383 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_22 = linear_102.view(1, -1, 4, 49)
        linear_102 = None
        q_34 = q_33.transpose(1, 2)
        q_33 = None
        k_23 = k_22.transpose(1, 2)
        k_22 = None
        v_23 = v_22.transpose(1, 2)
        v_22 = None
        q_35 = q_34.transpose(1, 2)
        q_34 = None
        linear_103 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_22 = linear_103.view(1, -1, 4, 49)
        linear_103 = None
        p_23 = p_22.transpose(1, 2)
        p_22 = None
        add_126 = (
            q_35
            + l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_11 = add_126.transpose(1, 2)
        add_126 = None
        add_127 = (
            q_35
            + l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_
        )
        q_35 = l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_11 = add_127.transpose(1, 2)
        add_127 = None
        transpose_144 = p_23.transpose(-2, -1)
        p_23 = None
        matrix_bd_22 = torch.matmul(q_with_bias_v_11, transpose_144)
        q_with_bias_v_11 = transpose_144 = None
        x_384 = torch._C._nn.pad(matrix_bd_22, (1, 0), "constant", None)
        matrix_bd_22 = None
        x_385 = x_384.view(1, 4, -1, 66)
        x_384 = None
        getitem_27 = x_385[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_385 = None
        x_386 = getitem_27.view(1, 4, 66, 131)
        getitem_27 = None
        transpose_145 = k_23.transpose(-2, -1)
        k_23 = None
        matrix_ac_11 = torch.matmul(q_with_bias_u_11, transpose_145)
        q_with_bias_u_11 = transpose_145 = None
        matrix_bd_23 = x_386[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_386 = None
        add_128 = matrix_ac_11 + matrix_bd_23
        matrix_ac_11 = matrix_bd_23 = None
        scores_22 = add_128 / 7.0
        add_128 = None
        mask_12 = att_mask_3.unsqueeze(1)
        scores_23 = scores_22.masked_fill(mask_12, -10000.0)
        scores_22 = None
        softmax_11 = torch.softmax(scores_23, dim=-1)
        scores_23 = None
        attn_11 = softmax_11.masked_fill(mask_12, 0.0)
        softmax_11 = mask_12 = None
        p_attn_11 = torch.nn.functional.dropout(attn_11, 0.1, False, False)
        attn_11 = None
        x_387 = torch.matmul(p_attn_11, v_23)
        p_attn_11 = v_23 = None
        transpose_146 = x_387.transpose(1, 2)
        x_387 = None
        x_388 = transpose_146.reshape(1, -1, 196)
        transpose_146 = None
        out_11 = torch._C._nn.linear(
            x_388,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_388 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_80 = torch.nn.functional.dropout(out_11, 0.1, False, False)
        out_11 = None
        x_389 = x_382 + dropout_80
        x_382 = dropout_80 = None
        x_390 = torch.nn.functional.layer_norm(
            x_389,
            (196,),
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_389 = l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_
        ) = None
        scale_45 = l_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_45 = l_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_68 = x_390 * scale_45
        scale_45 = None
        x_391 = mul_68 + bias_45
        mul_68 = bias_45 = None
        x_392 = torch._C._nn.linear(
            x_391,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_391 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_393 = torch.nn.functional.silu(x_392, inplace=False)
        x_392 = None
        x_394 = torch.nn.functional.dropout(x_393, 0.1, False, False)
        x_393 = None
        x_395 = torch._C._nn.linear(
            x_394,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_394 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_82 = torch.nn.functional.dropout(x_395, 0.1, False, False)
        x_395 = None
        mul_69 = dropout_82 * 1.0
        dropout_82 = None
        x_396 = x_390 + mul_69
        x_390 = mul_69 = None
        x_397 = torch.nn.functional.layer_norm(
            x_396,
            (196,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_396 = l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_46 = l_instance_modules_layers_modules_11_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_conv_scale_parameters_scale_ = None
        bias_46 = l_instance_modules_layers_modules_11_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_conv_scale_parameters_bias_ = None
        mul_70 = x_397 * scale_46
        scale_46 = None
        x_398 = mul_70 + bias_46
        mul_70 = bias_46 = None
        x_399 = x_398.transpose(1, 2)
        x_398 = None
        x_400 = torch.conv1d(
            x_399,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_399 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_401 = torch.nn.functional.silu(x_400, inplace=True)
        x_400 = None
        unsqueeze_27 = pad_mask_1.unsqueeze(1)
        x_402 = x_401.masked_fill(unsqueeze_27, 0.0)
        x_401 = unsqueeze_27 = None
        new_x_11 = torch._C._nn.pad(x_402, (15, 15), "constant", None)
        x_402 = None
        x_403 = torch.conv1d(
            new_x_11,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_11 = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_404 = torch.nn.functional.batch_norm(
            x_403,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_403 = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_405 = torch.nn.functional.silu(x_404, inplace=False)
        x_404 = None
        x_406 = torch.conv1d(
            x_405,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_405 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_407 = x_406.transpose(1, 2)
        x_406 = None
        dropout_83 = torch.nn.functional.dropout(x_407, 0.1, False, False)
        x_407 = None
        x_408 = x_397 + dropout_83
        x_397 = dropout_83 = None
        x_409 = torch.nn.functional.layer_norm(
            x_408,
            (196,),
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_408 = (
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_
        ) = None
        scale_47 = l_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_47 = l_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_11_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_71 = x_409 * scale_47
        scale_47 = None
        x_410 = mul_71 + bias_47
        mul_71 = bias_47 = None
        x_411 = torch._C._nn.linear(
            x_410,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_410 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_412 = torch.nn.functional.silu(x_411, inplace=False)
        x_411 = None
        x_413 = torch.nn.functional.dropout(x_412, 0.1, False, False)
        x_412 = None
        x_414 = torch._C._nn.linear(
            x_413,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_413 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_85 = torch.nn.functional.dropout(x_414, 0.1, False, False)
        x_414 = None
        mul_72 = dropout_85 * 1.0
        dropout_85 = None
        x_415 = x_409 + mul_72
        x_409 = mul_72 = None
        x_416 = torch.nn.functional.layer_norm(
            x_415,
            (196,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_415 = l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_48 = l_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_48 = l_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_73 = x_416 * scale_48
        scale_48 = None
        x_417 = mul_73 + bias_48
        mul_73 = bias_48 = None
        linear_109 = torch._C._nn.linear(
            x_417,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_36 = linear_109.view(1, -1, 4, 49)
        linear_109 = None
        linear_110 = torch._C._nn.linear(
            x_417,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_24 = linear_110.view(1, -1, 4, 49)
        linear_110 = None
        linear_111 = torch._C._nn.linear(
            x_417,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_417 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_24 = linear_111.view(1, -1, 4, 49)
        linear_111 = None
        q_37 = q_36.transpose(1, 2)
        q_36 = None
        k_25 = k_24.transpose(1, 2)
        k_24 = None
        v_25 = v_24.transpose(1, 2)
        v_24 = None
        q_38 = q_37.transpose(1, 2)
        q_37 = None
        linear_112 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_24 = linear_112.view(1, -1, 4, 49)
        linear_112 = None
        p_25 = p_24.transpose(1, 2)
        p_24 = None
        add_137 = (
            q_38
            + l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_12 = add_137.transpose(1, 2)
        add_137 = None
        add_138 = (
            q_38
            + l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_
        )
        q_38 = l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_12 = add_138.transpose(1, 2)
        add_138 = None
        transpose_156 = p_25.transpose(-2, -1)
        p_25 = None
        matrix_bd_24 = torch.matmul(q_with_bias_v_12, transpose_156)
        q_with_bias_v_12 = transpose_156 = None
        x_418 = torch._C._nn.pad(matrix_bd_24, (1, 0), "constant", None)
        matrix_bd_24 = None
        x_419 = x_418.view(1, 4, -1, 66)
        x_418 = None
        getitem_29 = x_419[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_419 = None
        x_420 = getitem_29.view(1, 4, 66, 131)
        getitem_29 = None
        transpose_157 = k_25.transpose(-2, -1)
        k_25 = None
        matrix_ac_12 = torch.matmul(q_with_bias_u_12, transpose_157)
        q_with_bias_u_12 = transpose_157 = None
        matrix_bd_25 = x_420[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_420 = None
        add_139 = matrix_ac_12 + matrix_bd_25
        matrix_ac_12 = matrix_bd_25 = None
        scores_24 = add_139 / 7.0
        add_139 = None
        mask_13 = att_mask_3.unsqueeze(1)
        scores_25 = scores_24.masked_fill(mask_13, -10000.0)
        scores_24 = None
        softmax_12 = torch.softmax(scores_25, dim=-1)
        scores_25 = None
        attn_12 = softmax_12.masked_fill(mask_13, 0.0)
        softmax_12 = mask_13 = None
        p_attn_12 = torch.nn.functional.dropout(attn_12, 0.1, False, False)
        attn_12 = None
        x_421 = torch.matmul(p_attn_12, v_25)
        p_attn_12 = v_25 = None
        transpose_158 = x_421.transpose(1, 2)
        x_421 = None
        x_422 = transpose_158.reshape(1, -1, 196)
        transpose_158 = None
        out_12 = torch._C._nn.linear(
            x_422,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_422 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_87 = torch.nn.functional.dropout(out_12, 0.1, False, False)
        out_12 = None
        x_423 = x_416 + dropout_87
        x_416 = dropout_87 = None
        x_424 = torch.nn.functional.layer_norm(
            x_423,
            (196,),
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_423 = l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_
        ) = None
        scale_49 = l_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_49 = l_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_74 = x_424 * scale_49
        scale_49 = None
        x_425 = mul_74 + bias_49
        mul_74 = bias_49 = None
        x_426 = torch._C._nn.linear(
            x_425,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_425 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_427 = torch.nn.functional.silu(x_426, inplace=False)
        x_426 = None
        x_428 = torch.nn.functional.dropout(x_427, 0.1, False, False)
        x_427 = None
        x_429 = torch._C._nn.linear(
            x_428,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_428 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_89 = torch.nn.functional.dropout(x_429, 0.1, False, False)
        x_429 = None
        mul_75 = dropout_89 * 1.0
        dropout_89 = None
        x_430 = x_424 + mul_75
        x_424 = mul_75 = None
        x_431 = torch.nn.functional.layer_norm(
            x_430,
            (196,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_430 = l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_50 = l_instance_modules_layers_modules_12_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_conv_scale_parameters_scale_ = None
        bias_50 = l_instance_modules_layers_modules_12_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_conv_scale_parameters_bias_ = None
        mul_76 = x_431 * scale_50
        scale_50 = None
        x_432 = mul_76 + bias_50
        mul_76 = bias_50 = None
        x_433 = x_432.transpose(1, 2)
        x_432 = None
        x_434 = torch.conv1d(
            x_433,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_433 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_435 = torch.nn.functional.silu(x_434, inplace=True)
        x_434 = None
        unsqueeze_29 = pad_mask_1.unsqueeze(1)
        x_436 = x_435.masked_fill(unsqueeze_29, 0.0)
        x_435 = unsqueeze_29 = None
        new_x_12 = torch._C._nn.pad(x_436, (15, 15), "constant", None)
        x_436 = None
        x_437 = torch.conv1d(
            new_x_12,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_12 = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_438 = torch.nn.functional.batch_norm(
            x_437,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_437 = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_439 = torch.nn.functional.silu(x_438, inplace=False)
        x_438 = None
        x_440 = torch.conv1d(
            x_439,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_439 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_441 = x_440.transpose(1, 2)
        x_440 = None
        dropout_90 = torch.nn.functional.dropout(x_441, 0.1, False, False)
        x_441 = None
        x_442 = x_431 + dropout_90
        x_431 = dropout_90 = None
        x_443 = torch.nn.functional.layer_norm(
            x_442,
            (196,),
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_442 = (
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_
        ) = None
        scale_51 = l_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_51 = l_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_12_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_77 = x_443 * scale_51
        scale_51 = None
        x_444 = mul_77 + bias_51
        mul_77 = bias_51 = None
        x_445 = torch._C._nn.linear(
            x_444,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_444 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_446 = torch.nn.functional.silu(x_445, inplace=False)
        x_445 = None
        x_447 = torch.nn.functional.dropout(x_446, 0.1, False, False)
        x_446 = None
        x_448 = torch._C._nn.linear(
            x_447,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_447 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_92 = torch.nn.functional.dropout(x_448, 0.1, False, False)
        x_448 = None
        mul_78 = dropout_92 * 1.0
        dropout_92 = None
        x_449 = x_443 + mul_78
        x_443 = mul_78 = None
        x_450 = torch.nn.functional.layer_norm(
            x_449,
            (196,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_449 = l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_52 = l_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_52 = l_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_79 = x_450 * scale_52
        scale_52 = None
        x_451 = mul_79 + bias_52
        mul_79 = bias_52 = None
        linear_118 = torch._C._nn.linear(
            x_451,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_39 = linear_118.view(1, -1, 4, 49)
        linear_118 = None
        linear_119 = torch._C._nn.linear(
            x_451,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_26 = linear_119.view(1, -1, 4, 49)
        linear_119 = None
        linear_120 = torch._C._nn.linear(
            x_451,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_451 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_26 = linear_120.view(1, -1, 4, 49)
        linear_120 = None
        q_40 = q_39.transpose(1, 2)
        q_39 = None
        k_27 = k_26.transpose(1, 2)
        k_26 = None
        v_27 = v_26.transpose(1, 2)
        v_26 = None
        q_41 = q_40.transpose(1, 2)
        q_40 = None
        linear_121 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_26 = linear_121.view(1, -1, 4, 49)
        linear_121 = None
        p_27 = p_26.transpose(1, 2)
        p_26 = None
        add_148 = (
            q_41
            + l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_13 = add_148.transpose(1, 2)
        add_148 = None
        add_149 = (
            q_41
            + l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_
        )
        q_41 = l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_13 = add_149.transpose(1, 2)
        add_149 = None
        transpose_168 = p_27.transpose(-2, -1)
        p_27 = None
        matrix_bd_26 = torch.matmul(q_with_bias_v_13, transpose_168)
        q_with_bias_v_13 = transpose_168 = None
        x_452 = torch._C._nn.pad(matrix_bd_26, (1, 0), "constant", None)
        matrix_bd_26 = None
        x_453 = x_452.view(1, 4, -1, 66)
        x_452 = None
        getitem_31 = x_453[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_453 = None
        x_454 = getitem_31.view(1, 4, 66, 131)
        getitem_31 = None
        transpose_169 = k_27.transpose(-2, -1)
        k_27 = None
        matrix_ac_13 = torch.matmul(q_with_bias_u_13, transpose_169)
        q_with_bias_u_13 = transpose_169 = None
        matrix_bd_27 = x_454[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_454 = None
        add_150 = matrix_ac_13 + matrix_bd_27
        matrix_ac_13 = matrix_bd_27 = None
        scores_26 = add_150 / 7.0
        add_150 = None
        mask_14 = att_mask_3.unsqueeze(1)
        scores_27 = scores_26.masked_fill(mask_14, -10000.0)
        scores_26 = None
        softmax_13 = torch.softmax(scores_27, dim=-1)
        scores_27 = None
        attn_13 = softmax_13.masked_fill(mask_14, 0.0)
        softmax_13 = mask_14 = None
        p_attn_13 = torch.nn.functional.dropout(attn_13, 0.1, False, False)
        attn_13 = None
        x_455 = torch.matmul(p_attn_13, v_27)
        p_attn_13 = v_27 = None
        transpose_170 = x_455.transpose(1, 2)
        x_455 = None
        x_456 = transpose_170.reshape(1, -1, 196)
        transpose_170 = None
        out_13 = torch._C._nn.linear(
            x_456,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_456 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_94 = torch.nn.functional.dropout(out_13, 0.1, False, False)
        out_13 = None
        x_457 = x_450 + dropout_94
        x_450 = dropout_94 = None
        x_458 = torch.nn.functional.layer_norm(
            x_457,
            (196,),
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_457 = l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_
        ) = None
        scale_53 = l_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_53 = l_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_80 = x_458 * scale_53
        scale_53 = None
        x_459 = mul_80 + bias_53
        mul_80 = bias_53 = None
        x_460 = torch._C._nn.linear(
            x_459,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_459 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_461 = torch.nn.functional.silu(x_460, inplace=False)
        x_460 = None
        x_462 = torch.nn.functional.dropout(x_461, 0.1, False, False)
        x_461 = None
        x_463 = torch._C._nn.linear(
            x_462,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_462 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_96 = torch.nn.functional.dropout(x_463, 0.1, False, False)
        x_463 = None
        mul_81 = dropout_96 * 1.0
        dropout_96 = None
        x_464 = x_458 + mul_81
        x_458 = mul_81 = None
        x_465 = torch.nn.functional.layer_norm(
            x_464,
            (196,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_464 = l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_54 = l_instance_modules_layers_modules_13_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_conv_scale_parameters_scale_ = None
        bias_54 = l_instance_modules_layers_modules_13_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_conv_scale_parameters_bias_ = None
        mul_82 = x_465 * scale_54
        scale_54 = None
        x_466 = mul_82 + bias_54
        mul_82 = bias_54 = None
        x_467 = x_466.transpose(1, 2)
        x_466 = None
        x_468 = torch.conv1d(
            x_467,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_467 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_469 = torch.nn.functional.silu(x_468, inplace=True)
        x_468 = None
        unsqueeze_31 = pad_mask_1.unsqueeze(1)
        x_470 = x_469.masked_fill(unsqueeze_31, 0.0)
        x_469 = unsqueeze_31 = None
        new_x_13 = torch._C._nn.pad(x_470, (15, 15), "constant", None)
        x_470 = None
        x_471 = torch.conv1d(
            new_x_13,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_13 = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_472 = torch.nn.functional.batch_norm(
            x_471,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_471 = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_473 = torch.nn.functional.silu(x_472, inplace=False)
        x_472 = None
        x_474 = torch.conv1d(
            x_473,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_473 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_475 = x_474.transpose(1, 2)
        x_474 = None
        dropout_97 = torch.nn.functional.dropout(x_475, 0.1, False, False)
        x_475 = None
        x_476 = x_465 + dropout_97
        x_465 = dropout_97 = None
        x_477 = torch.nn.functional.layer_norm(
            x_476,
            (196,),
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_476 = (
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_
        ) = None
        scale_55 = l_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_55 = l_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_13_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_83 = x_477 * scale_55
        scale_55 = None
        x_478 = mul_83 + bias_55
        mul_83 = bias_55 = None
        x_479 = torch._C._nn.linear(
            x_478,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_478 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_480 = torch.nn.functional.silu(x_479, inplace=False)
        x_479 = None
        x_481 = torch.nn.functional.dropout(x_480, 0.1, False, False)
        x_480 = None
        x_482 = torch._C._nn.linear(
            x_481,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_481 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_99 = torch.nn.functional.dropout(x_482, 0.1, False, False)
        x_482 = None
        mul_84 = dropout_99 * 1.0
        dropout_99 = None
        x_483 = x_477 + mul_84
        x_477 = mul_84 = None
        x_484 = torch.nn.functional.layer_norm(
            x_483,
            (196,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_483 = l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_56 = l_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_56 = l_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_85 = x_484 * scale_56
        scale_56 = None
        x_485 = mul_85 + bias_56
        mul_85 = bias_56 = None
        linear_127 = torch._C._nn.linear(
            x_485,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_42 = linear_127.view(1, -1, 4, 49)
        linear_127 = None
        linear_128 = torch._C._nn.linear(
            x_485,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_28 = linear_128.view(1, -1, 4, 49)
        linear_128 = None
        linear_129 = torch._C._nn.linear(
            x_485,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_485 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_28 = linear_129.view(1, -1, 4, 49)
        linear_129 = None
        q_43 = q_42.transpose(1, 2)
        q_42 = None
        k_29 = k_28.transpose(1, 2)
        k_28 = None
        v_29 = v_28.transpose(1, 2)
        v_28 = None
        q_44 = q_43.transpose(1, 2)
        q_43 = None
        linear_130 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_28 = linear_130.view(1, -1, 4, 49)
        linear_130 = None
        p_29 = p_28.transpose(1, 2)
        p_28 = None
        add_159 = (
            q_44
            + l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_14 = add_159.transpose(1, 2)
        add_159 = None
        add_160 = (
            q_44
            + l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_
        )
        q_44 = l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_14 = add_160.transpose(1, 2)
        add_160 = None
        transpose_180 = p_29.transpose(-2, -1)
        p_29 = None
        matrix_bd_28 = torch.matmul(q_with_bias_v_14, transpose_180)
        q_with_bias_v_14 = transpose_180 = None
        x_486 = torch._C._nn.pad(matrix_bd_28, (1, 0), "constant", None)
        matrix_bd_28 = None
        x_487 = x_486.view(1, 4, -1, 66)
        x_486 = None
        getitem_33 = x_487[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_487 = None
        x_488 = getitem_33.view(1, 4, 66, 131)
        getitem_33 = None
        transpose_181 = k_29.transpose(-2, -1)
        k_29 = None
        matrix_ac_14 = torch.matmul(q_with_bias_u_14, transpose_181)
        q_with_bias_u_14 = transpose_181 = None
        matrix_bd_29 = x_488[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_488 = None
        add_161 = matrix_ac_14 + matrix_bd_29
        matrix_ac_14 = matrix_bd_29 = None
        scores_28 = add_161 / 7.0
        add_161 = None
        mask_15 = att_mask_3.unsqueeze(1)
        scores_29 = scores_28.masked_fill(mask_15, -10000.0)
        scores_28 = None
        softmax_14 = torch.softmax(scores_29, dim=-1)
        scores_29 = None
        attn_14 = softmax_14.masked_fill(mask_15, 0.0)
        softmax_14 = mask_15 = None
        p_attn_14 = torch.nn.functional.dropout(attn_14, 0.1, False, False)
        attn_14 = None
        x_489 = torch.matmul(p_attn_14, v_29)
        p_attn_14 = v_29 = None
        transpose_182 = x_489.transpose(1, 2)
        x_489 = None
        x_490 = transpose_182.reshape(1, -1, 196)
        transpose_182 = None
        out_14 = torch._C._nn.linear(
            x_490,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_490 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_101 = torch.nn.functional.dropout(out_14, 0.1, False, False)
        out_14 = None
        x_491 = x_484 + dropout_101
        x_484 = dropout_101 = None
        x_492 = torch.nn.functional.layer_norm(
            x_491,
            (196,),
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_491 = l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_
        ) = None
        scale_57 = l_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_57 = l_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_86 = x_492 * scale_57
        scale_57 = None
        x_493 = mul_86 + bias_57
        mul_86 = bias_57 = None
        x_494 = torch._C._nn.linear(
            x_493,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_493 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_495 = torch.nn.functional.silu(x_494, inplace=False)
        x_494 = None
        x_496 = torch.nn.functional.dropout(x_495, 0.1, False, False)
        x_495 = None
        x_497 = torch._C._nn.linear(
            x_496,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_496 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_103 = torch.nn.functional.dropout(x_497, 0.1, False, False)
        x_497 = None
        mul_87 = dropout_103 * 1.0
        dropout_103 = None
        x_498 = x_492 + mul_87
        x_492 = mul_87 = None
        x_499 = torch.nn.functional.layer_norm(
            x_498,
            (196,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_498 = l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_58 = l_instance_modules_layers_modules_14_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_conv_scale_parameters_scale_ = None
        bias_58 = l_instance_modules_layers_modules_14_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_conv_scale_parameters_bias_ = None
        mul_88 = x_499 * scale_58
        scale_58 = None
        x_500 = mul_88 + bias_58
        mul_88 = bias_58 = None
        x_501 = x_500.transpose(1, 2)
        x_500 = None
        x_502 = torch.conv1d(
            x_501,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_501 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_503 = torch.nn.functional.silu(x_502, inplace=True)
        x_502 = None
        unsqueeze_33 = pad_mask_1.unsqueeze(1)
        x_504 = x_503.masked_fill(unsqueeze_33, 0.0)
        x_503 = unsqueeze_33 = None
        new_x_14 = torch._C._nn.pad(x_504, (15, 15), "constant", None)
        x_504 = None
        x_505 = torch.conv1d(
            new_x_14,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_14 = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_506 = torch.nn.functional.batch_norm(
            x_505,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_505 = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_507 = torch.nn.functional.silu(x_506, inplace=False)
        x_506 = None
        x_508 = torch.conv1d(
            x_507,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_507 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_509 = x_508.transpose(1, 2)
        x_508 = None
        dropout_104 = torch.nn.functional.dropout(x_509, 0.1, False, False)
        x_509 = None
        x_510 = x_499 + dropout_104
        x_499 = dropout_104 = None
        x_511 = torch.nn.functional.layer_norm(
            x_510,
            (196,),
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_510 = (
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_
        ) = None
        scale_59 = l_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_59 = l_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_14_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_89 = x_511 * scale_59
        scale_59 = None
        x_512 = mul_89 + bias_59
        mul_89 = bias_59 = None
        x_513 = torch._C._nn.linear(
            x_512,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_512 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_514 = torch.nn.functional.silu(x_513, inplace=False)
        x_513 = None
        x_515 = torch.nn.functional.dropout(x_514, 0.1, False, False)
        x_514 = None
        x_516 = torch._C._nn.linear(
            x_515,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_515 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_106 = torch.nn.functional.dropout(x_516, 0.1, False, False)
        x_516 = None
        mul_90 = dropout_106 * 1.0
        dropout_106 = None
        x_517 = x_511 + mul_90
        x_511 = mul_90 = None
        x_518 = torch.nn.functional.layer_norm(
            x_517,
            (196,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_517 = l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_60 = l_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_60 = l_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_91 = x_518 * scale_60
        scale_60 = None
        x_519 = mul_91 + bias_60
        mul_91 = bias_60 = None
        linear_136 = torch._C._nn.linear(
            x_519,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_45 = linear_136.view(1, -1, 4, 49)
        linear_136 = None
        linear_137 = torch._C._nn.linear(
            x_519,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_30 = linear_137.view(1, -1, 4, 49)
        linear_137 = None
        linear_138 = torch._C._nn.linear(
            x_519,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_519 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_30 = linear_138.view(1, -1, 4, 49)
        linear_138 = None
        q_46 = q_45.transpose(1, 2)
        q_45 = None
        k_31 = k_30.transpose(1, 2)
        k_30 = None
        v_31 = v_30.transpose(1, 2)
        v_30 = None
        q_47 = q_46.transpose(1, 2)
        q_46 = None
        linear_139 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_30 = linear_139.view(1, -1, 4, 49)
        linear_139 = None
        p_31 = p_30.transpose(1, 2)
        p_30 = None
        add_170 = (
            q_47
            + l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_15 = add_170.transpose(1, 2)
        add_170 = None
        add_171 = (
            q_47
            + l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_
        )
        q_47 = l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_15 = add_171.transpose(1, 2)
        add_171 = None
        transpose_192 = p_31.transpose(-2, -1)
        p_31 = None
        matrix_bd_30 = torch.matmul(q_with_bias_v_15, transpose_192)
        q_with_bias_v_15 = transpose_192 = None
        x_520 = torch._C._nn.pad(matrix_bd_30, (1, 0), "constant", None)
        matrix_bd_30 = None
        x_521 = x_520.view(1, 4, -1, 66)
        x_520 = None
        getitem_35 = x_521[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_521 = None
        x_522 = getitem_35.view(1, 4, 66, 131)
        getitem_35 = None
        transpose_193 = k_31.transpose(-2, -1)
        k_31 = None
        matrix_ac_15 = torch.matmul(q_with_bias_u_15, transpose_193)
        q_with_bias_u_15 = transpose_193 = None
        matrix_bd_31 = x_522[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_522 = None
        add_172 = matrix_ac_15 + matrix_bd_31
        matrix_ac_15 = matrix_bd_31 = None
        scores_30 = add_172 / 7.0
        add_172 = None
        mask_16 = att_mask_3.unsqueeze(1)
        scores_31 = scores_30.masked_fill(mask_16, -10000.0)
        scores_30 = None
        softmax_15 = torch.softmax(scores_31, dim=-1)
        scores_31 = None
        attn_15 = softmax_15.masked_fill(mask_16, 0.0)
        softmax_15 = mask_16 = None
        p_attn_15 = torch.nn.functional.dropout(attn_15, 0.1, False, False)
        attn_15 = None
        x_523 = torch.matmul(p_attn_15, v_31)
        p_attn_15 = v_31 = None
        transpose_194 = x_523.transpose(1, 2)
        x_523 = None
        x_524 = transpose_194.reshape(1, -1, 196)
        transpose_194 = None
        out_15 = torch._C._nn.linear(
            x_524,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_524 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_108 = torch.nn.functional.dropout(out_15, 0.1, False, False)
        out_15 = None
        x_525 = x_518 + dropout_108
        x_518 = dropout_108 = None
        x_526 = torch.nn.functional.layer_norm(
            x_525,
            (196,),
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_525 = l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_
        ) = None
        scale_61 = l_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_61 = l_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_92 = x_526 * scale_61
        scale_61 = None
        x_527 = mul_92 + bias_61
        mul_92 = bias_61 = None
        x_528 = torch._C._nn.linear(
            x_527,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_527 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_529 = torch.nn.functional.silu(x_528, inplace=False)
        x_528 = None
        x_530 = torch.nn.functional.dropout(x_529, 0.1, False, False)
        x_529 = None
        x_531 = torch._C._nn.linear(
            x_530,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_530 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_110 = torch.nn.functional.dropout(x_531, 0.1, False, False)
        x_531 = None
        mul_93 = dropout_110 * 1.0
        dropout_110 = None
        x_532 = x_526 + mul_93
        x_526 = mul_93 = None
        x_533 = torch.nn.functional.layer_norm(
            x_532,
            (196,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_532 = l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_62 = l_instance_modules_layers_modules_15_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_conv_scale_parameters_scale_ = None
        bias_62 = l_instance_modules_layers_modules_15_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_conv_scale_parameters_bias_ = None
        mul_94 = x_533 * scale_62
        scale_62 = None
        x_534 = mul_94 + bias_62
        mul_94 = bias_62 = None
        x_535 = x_534.transpose(1, 2)
        x_534 = None
        x_536 = torch.conv1d(
            x_535,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_535 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_537 = torch.nn.functional.silu(x_536, inplace=True)
        x_536 = None
        unsqueeze_35 = pad_mask_1.unsqueeze(1)
        x_538 = x_537.masked_fill(unsqueeze_35, 0.0)
        x_537 = unsqueeze_35 = None
        new_x_15 = torch._C._nn.pad(x_538, (15, 15), "constant", None)
        x_538 = None
        x_539 = torch.conv1d(
            new_x_15,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_15 = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_540 = torch.nn.functional.batch_norm(
            x_539,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_539 = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_541 = torch.nn.functional.silu(x_540, inplace=False)
        x_540 = None
        x_542 = torch.conv1d(
            x_541,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_541 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_543 = x_542.transpose(1, 2)
        x_542 = None
        dropout_111 = torch.nn.functional.dropout(x_543, 0.1, False, False)
        x_543 = None
        x_544 = x_533 + dropout_111
        x_533 = dropout_111 = None
        x_545 = torch.nn.functional.layer_norm(
            x_544,
            (196,),
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_544 = (
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_
        ) = None
        scale_63 = l_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_63 = l_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_15_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_95 = x_545 * scale_63
        scale_63 = None
        x_546 = mul_95 + bias_63
        mul_95 = bias_63 = None
        x_547 = torch._C._nn.linear(
            x_546,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_546 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_548 = torch.nn.functional.silu(x_547, inplace=False)
        x_547 = None
        x_549 = torch.nn.functional.dropout(x_548, 0.1, False, False)
        x_548 = None
        x_550 = torch._C._nn.linear(
            x_549,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_549 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_113 = torch.nn.functional.dropout(x_550, 0.1, False, False)
        x_550 = None
        mul_96 = dropout_113 * 1.0
        dropout_113 = None
        x_551 = x_545 + mul_96
        x_545 = mul_96 = None
        x_552 = torch.nn.functional.layer_norm(
            x_551,
            (196,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_551 = l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_ = (None)
        scale_64 = l_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_64 = l_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_97 = x_552 * scale_64
        scale_64 = None
        x_553 = mul_97 + bias_64
        mul_97 = bias_64 = None
        linear_145 = torch._C._nn.linear(
            x_553,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_48 = linear_145.view(1, -1, 4, 49)
        linear_145 = None
        linear_146 = torch._C._nn.linear(
            x_553,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_32 = linear_146.view(1, -1, 4, 49)
        linear_146 = None
        linear_147 = torch._C._nn.linear(
            x_553,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_553 = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_32 = linear_147.view(1, -1, 4, 49)
        linear_147 = None
        q_49 = q_48.transpose(1, 2)
        q_48 = None
        k_33 = k_32.transpose(1, 2)
        k_32 = None
        v_33 = v_32.transpose(1, 2)
        v_32 = None
        q_50 = q_49.transpose(1, 2)
        q_49 = None
        linear_148 = torch._C._nn.linear(
            pos_emb_1,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        pos_emb_1 = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_ = (None)
        p_32 = linear_148.view(1, -1, 4, 49)
        linear_148 = None
        p_33 = p_32.transpose(1, 2)
        p_32 = None
        add_181 = (
            q_50
            + l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_16 = add_181.transpose(1, 2)
        add_181 = None
        add_182 = (
            q_50
            + l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_
        )
        q_50 = l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_16 = add_182.transpose(1, 2)
        add_182 = None
        transpose_204 = p_33.transpose(-2, -1)
        p_33 = None
        matrix_bd_32 = torch.matmul(q_with_bias_v_16, transpose_204)
        q_with_bias_v_16 = transpose_204 = None
        x_554 = torch._C._nn.pad(matrix_bd_32, (1, 0), "constant", None)
        matrix_bd_32 = None
        x_555 = x_554.view(1, 4, -1, 66)
        x_554 = None
        getitem_37 = x_555[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_555 = None
        x_556 = getitem_37.view(1, 4, 66, 131)
        getitem_37 = None
        transpose_205 = k_33.transpose(-2, -1)
        k_33 = None
        matrix_ac_16 = torch.matmul(q_with_bias_u_16, transpose_205)
        q_with_bias_u_16 = transpose_205 = None
        matrix_bd_33 = x_556[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_556 = None
        add_183 = matrix_ac_16 + matrix_bd_33
        matrix_ac_16 = matrix_bd_33 = None
        scores_32 = add_183 / 7.0
        add_183 = None
        mask_17 = att_mask_3.unsqueeze(1)
        att_mask_3 = None
        scores_33 = scores_32.masked_fill(mask_17, -10000.0)
        scores_32 = None
        softmax_16 = torch.softmax(scores_33, dim=-1)
        scores_33 = None
        attn_16 = softmax_16.masked_fill(mask_17, 0.0)
        softmax_16 = mask_17 = None
        p_attn_16 = torch.nn.functional.dropout(attn_16, 0.1, False, False)
        attn_16 = None
        x_557 = torch.matmul(p_attn_16, v_33)
        p_attn_16 = v_33 = None
        transpose_206 = x_557.transpose(1, 2)
        x_557 = None
        x_558 = transpose_206.reshape(1, -1, 196)
        transpose_206 = None
        out_16 = torch._C._nn.linear(
            x_558,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_558 = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_115 = torch.nn.functional.dropout(out_16, 0.1, False, False)
        out_16 = None
        x_559 = x_552 + dropout_115
        x_552 = dropout_115 = None
        x_560 = torch.nn.functional.layer_norm(
            x_559,
            (196,),
            l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_559 = l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_
        ) = None
        scale_65 = l_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_65 = l_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_98 = x_560 * scale_65
        scale_65 = None
        x_561 = mul_98 + bias_65
        mul_98 = bias_65 = None
        x_562 = torch._C._nn.linear(
            x_561,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_561 = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_563 = torch.nn.functional.silu(x_562, inplace=False)
        x_562 = None
        x_564 = torch.nn.functional.dropout(x_563, 0.1, False, False)
        x_563 = None
        x_565 = torch._C._nn.linear(
            x_564,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_564 = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_117 = torch.nn.functional.dropout(x_565, 0.1, False, False)
        x_565 = None
        mul_99 = dropout_117 * 1.0
        dropout_117 = None
        x_566 = x_560 + mul_99
        x_560 = mul_99 = None
        x_567 = torch.nn.functional.layer_norm(
            x_566,
            (196,),
            l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_566 = l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_66 = l_instance_modules_layers_modules_16_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_conv_scale_parameters_scale_ = None
        bias_66 = l_instance_modules_layers_modules_16_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_conv_scale_parameters_bias_ = None
        mul_100 = x_567 * scale_66
        scale_66 = None
        x_568 = mul_100 + bias_66
        mul_100 = bias_66 = None
        x_569 = x_568.transpose(1, 2)
        x_568 = None
        x_570 = torch.conv1d(
            x_569,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_569 = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_571 = torch.nn.functional.silu(x_570, inplace=True)
        x_570 = None
        unsqueeze_37 = pad_mask_1.unsqueeze(1)
        pad_mask_1 = None
        x_572 = x_571.masked_fill(unsqueeze_37, 0.0)
        x_571 = unsqueeze_37 = None
        new_x_16 = torch._C._nn.pad(x_572, (15, 15), "constant", None)
        x_572 = None
        x_573 = torch.conv1d(
            new_x_16,
            l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_16 = l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_574 = torch.nn.functional.batch_norm(
            x_573,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_573 = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_575 = torch.nn.functional.silu(x_574, inplace=False)
        x_574 = None
        x_576 = torch.conv1d(
            x_575,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_575 = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_577 = x_576.transpose(1, 2)
        x_576 = None
        dropout_118 = torch.nn.functional.dropout(x_577, 0.1, False, False)
        x_577 = None
        x_578 = x_567 + dropout_118
        x_567 = dropout_118 = None
        x_579 = torch.nn.functional.layer_norm(
            x_578,
            (196,),
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_578 = (
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_
        ) = None
        scale_67 = l_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_67 = l_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_16_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_101 = x_579 * scale_67
        scale_67 = None
        x_580 = mul_101 + bias_67
        mul_101 = bias_67 = None
        x_581 = torch._C._nn.linear(
            x_580,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_580 = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_582 = torch.nn.functional.silu(x_581, inplace=False)
        x_581 = None
        x_583 = torch.nn.functional.dropout(x_582, 0.1, False, False)
        x_582 = None
        x_584 = torch._C._nn.linear(
            x_583,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_583 = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_120 = torch.nn.functional.dropout(x_584, 0.1, False, False)
        x_584 = None
        mul_102 = dropout_120 * 1.0
        dropout_120 = None
        x_585 = x_579 + mul_102
        x_579 = mul_102 = None
        x_586 = torch.nn.functional.layer_norm(
            x_585,
            (196,),
            l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_585 = l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_ = (None)
        audio_signal_3 = torch.repeat_interleave(x_586, repeats=2, dim=1)
        x_586 = None
        audio_signal_4 = audio_signal_3[
            (slice(None, None, None), slice(None, 131, None), slice(None, None, None))
        ]
        audio_signal_3 = None
        audio_signal_5 = torch._C._nn.linear(
            audio_signal_4,
            l_instance_modules_time_recovery_layer_parameters_weight_,
            l_instance_modules_time_recovery_layer_parameters_bias_,
        )
        audio_signal_4 = (
            l_instance_modules_time_recovery_layer_parameters_weight_
        ) = l_instance_modules_time_recovery_layer_parameters_bias_ = None
        audio_signal_6 = x_274 + audio_signal_5
        x_274 = audio_signal_5 = None
        scale_68 = l_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_scale_ = (
            None
        )
        bias_68 = l_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_self_attn_scale_parameters_bias_ = (
            None
        )
        mul_103 = audio_signal_6 * scale_68
        scale_68 = None
        x_587 = mul_103 + bias_68
        mul_103 = bias_68 = None
        linear_155 = torch._C._nn.linear(
            x_587,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_51 = linear_155.view(1, -1, 4, 49)
        linear_155 = None
        linear_156 = torch._C._nn.linear(
            x_587,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_34 = linear_156.view(1, -1, 4, 49)
        linear_156 = None
        linear_157 = torch._C._nn.linear(
            x_587,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_587 = l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_34 = linear_157.view(1, -1, 4, 49)
        linear_157 = None
        q_52 = q_51.transpose(1, 2)
        q_51 = None
        k_35 = k_34.transpose(1, 2)
        k_34 = None
        v_35 = v_34.transpose(1, 2)
        v_34 = None
        q_53 = q_52.transpose(1, 2)
        q_52 = None
        linear_158 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        pos_emb = l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_pos_parameters_weight_ = (None)
        p_34 = linear_158.view(1, -1, 4, 49)
        linear_158 = None
        p_35 = p_34.transpose(1, 2)
        p_34 = None
        add_193 = (
            q_53
            + l_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_17 = add_193.transpose(1, 2)
        add_193 = None
        add_194 = (
            q_53
            + l_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_v_
        )
        q_53 = l_instance_modules_layers_modules_17_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_17 = add_194.transpose(1, 2)
        add_194 = None
        transpose_216 = p_35.transpose(-2, -1)
        p_35 = None
        matrix_bd_34 = torch.matmul(q_with_bias_v_17, transpose_216)
        q_with_bias_v_17 = transpose_216 = None
        x_588 = torch._C._nn.pad(matrix_bd_34, (1, 0), "constant", None)
        matrix_bd_34 = None
        x_589 = x_588.view(1, 4, -1, 131)
        x_588 = None
        getitem_40 = x_589[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_589 = None
        x_590 = getitem_40.view(1, 4, 131, 261)
        getitem_40 = None
        transpose_217 = k_35.transpose(-2, -1)
        k_35 = None
        matrix_ac_17 = torch.matmul(q_with_bias_u_17, transpose_217)
        q_with_bias_u_17 = transpose_217 = None
        matrix_bd_35 = x_590[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_590 = None
        add_195 = matrix_ac_17 + matrix_bd_35
        matrix_ac_17 = matrix_bd_35 = None
        scores_34 = add_195 / 7.0
        add_195 = None
        mask_18 = att_mask_2.unsqueeze(1)
        att_mask_2 = None
        scores_35 = scores_34.masked_fill(mask_18, -10000.0)
        scores_34 = None
        softmax_17 = torch.softmax(scores_35, dim=-1)
        scores_35 = None
        attn_17 = softmax_17.masked_fill(mask_18, 0.0)
        softmax_17 = mask_18 = None
        p_attn_17 = torch.nn.functional.dropout(attn_17, 0.1, False, False)
        attn_17 = None
        x_591 = torch.matmul(p_attn_17, v_35)
        p_attn_17 = v_35 = None
        transpose_218 = x_591.transpose(1, 2)
        x_591 = None
        x_592 = transpose_218.reshape(1, -1, 196)
        transpose_218 = None
        out_17 = torch._C._nn.linear(
            x_592,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_592 = l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_17_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_122 = torch.nn.functional.dropout(out_17, 0.1, False, False)
        out_17 = None
        x_593 = audio_signal_6 + dropout_122
        audio_signal_6 = dropout_122 = None
        x_594 = torch.nn.functional.layer_norm(
            x_593,
            (196,),
            l_instance_modules_layers_modules_17_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        x_593 = l_instance_modules_layers_modules_17_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_17_modules_norm_self_att_parameters_bias_
        ) = None
        scale_69 = l_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_scale_ = (
            None
        )
        bias_69 = l_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_feed_forward1_scale_parameters_bias_ = (
            None
        )
        mul_104 = x_594 * scale_69
        scale_69 = None
        x_595 = mul_104 + bias_69
        mul_104 = bias_69 = None
        x_596 = torch._C._nn.linear(
            x_595,
            l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_595 = l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_597 = torch.nn.functional.silu(x_596, inplace=False)
        x_596 = None
        x_598 = torch.nn.functional.dropout(x_597, 0.1, False, False)
        x_597 = None
        x_599 = torch._C._nn.linear(
            x_598,
            l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_598 = l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_17_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_124 = torch.nn.functional.dropout(x_599, 0.1, False, False)
        x_599 = None
        mul_105 = dropout_124 * 1.0
        dropout_124 = None
        x_600 = x_594 + mul_105
        x_594 = mul_105 = None
        x_601 = torch.nn.functional.layer_norm(
            x_600,
            (196,),
            l_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        x_600 = l_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_17_modules_norm_feed_forward1_parameters_bias_ = (None)
        scale_70 = l_instance_modules_layers_modules_17_modules_conv_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_conv_scale_parameters_scale_ = None
        bias_70 = l_instance_modules_layers_modules_17_modules_conv_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_conv_scale_parameters_bias_ = None
        mul_106 = x_601 * scale_70
        scale_70 = None
        x_602 = mul_106 + bias_70
        mul_106 = bias_70 = None
        x_603 = x_602.transpose(1, 2)
        x_602 = None
        x_604 = torch.conv1d(
            x_603,
            l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_603 = l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_605 = torch.nn.functional.silu(x_604, inplace=True)
        x_604 = None
        unsqueeze_39 = pad_mask.unsqueeze(1)
        pad_mask = None
        x_606 = x_605.masked_fill(unsqueeze_39, 0.0)
        x_605 = unsqueeze_39 = None
        new_x_17 = torch._C._nn.pad(x_606, (15, 15), "constant", None)
        x_606 = None
        x_607 = torch.conv1d(
            new_x_17,
            l_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            392,
        )
        new_x_17 = l_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_17_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_608 = torch.nn.functional.batch_norm(
            x_607,
            l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_607 = l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_17_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_609 = torch.nn.functional.silu(x_608, inplace=False)
        x_608 = None
        x_610 = torch.conv1d(
            x_609,
            l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_609 = l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_17_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_611 = x_610.transpose(1, 2)
        x_610 = None
        dropout_125 = torch.nn.functional.dropout(x_611, 0.1, False, False)
        x_611 = None
        x_612 = x_601 + dropout_125
        x_601 = dropout_125 = None
        x_613 = torch.nn.functional.layer_norm(
            x_612,
            (196,),
            l_instance_modules_layers_modules_17_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        x_612 = (
            l_instance_modules_layers_modules_17_modules_norm_conv_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_17_modules_norm_conv_parameters_bias_
        ) = None
        scale_71 = l_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_scale_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_scale_ = (
            None
        )
        bias_71 = l_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_bias_.view(
            1, 1, -1
        )
        l_instance_modules_layers_modules_17_modules_feed_forward2_scale_parameters_bias_ = (
            None
        )
        mul_107 = x_613 * scale_71
        scale_71 = None
        x_614 = mul_107 + bias_71
        mul_107 = bias_71 = None
        x_615 = torch._C._nn.linear(
            x_614,
            l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_614 = l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_616 = torch.nn.functional.silu(x_615, inplace=False)
        x_615 = None
        x_617 = torch.nn.functional.dropout(x_616, 0.1, False, False)
        x_616 = None
        x_618 = torch._C._nn.linear(
            x_617,
            l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_617 = l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_17_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_127 = torch.nn.functional.dropout(x_618, 0.1, False, False)
        x_618 = None
        mul_108 = dropout_127 * 1.0
        dropout_127 = None
        x_619 = x_613 + mul_108
        x_613 = mul_108 = None
        x_620 = torch.nn.functional.layer_norm(
            x_619,
            (196,),
            l_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        x_619 = l_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_17_modules_norm_feed_forward2_parameters_bias_ = (None)
        audio_signal_7 = torch.transpose(x_620, 1, 2)
        x_620 = None
        return (audio_signal_7, lengths_4)
