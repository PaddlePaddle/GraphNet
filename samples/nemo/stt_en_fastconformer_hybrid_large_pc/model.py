import torch

from torch import device


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
        L_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_pre_encode_modules_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_pos_enc_buffers_pe_: torch.Tensor,
        L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_out_parameters_weight_: torch.nn.parameter.Parameter,
        L_instance_modules_layers_modules_16_modules_norm_out_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_ = (
            L_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_
        )
        l_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_ = (
            L_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_
        )
        l_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_ = (
            L_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_
        )
        l_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_ = (
            L_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_
        )
        l_instance_modules_pre_encode_modules_out_parameters_weight_ = (
            L_instance_modules_pre_encode_modules_out_parameters_weight_
        )
        l_instance_modules_pre_encode_modules_out_parameters_bias_ = (
            L_instance_modules_pre_encode_modules_out_parameters_bias_
        )
        l_instance_modules_pos_enc_buffers_pe_ = L_instance_modules_pos_enc_buffers_pe_
        l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_ = (
            L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_
        )
        l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_
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
        l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_
        )
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
        l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_
        )
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
        l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_
        )
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
        l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_
        )
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
        l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_
        )
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
        l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_
        )
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
        l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_
        )
        l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_
        l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_ = L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_
        l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_ = (
            L_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_
        )
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
        l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_ = (
            L_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_
        )
        l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_ = (
            L_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_
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
        l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_
        l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_ = L_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_
        l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_ = L_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_
        l_instance_modules_layers_modules_16_modules_norm_out_parameters_weight_ = (
            L_instance_modules_layers_modules_16_modules_norm_out_parameters_weight_
        )
        l_instance_modules_layers_modules_16_modules_norm_out_parameters_bias_ = (
            L_instance_modules_layers_modules_16_modules_norm_out_parameters_bias_
        )
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
        to_2 = lengths_3.to(dtype=torch.float32)
        lengths_3 = None
        add_4 = to_2 + -1
        to_2 = None
        div_2 = torch.div(add_4, 2)
        add_4 = None
        lengths_4 = div_2 + 1.0
        div_2 = None
        lengths_5 = torch.floor(lengths_4)
        lengths_4 = None
        lengths_6 = lengths_5.to(dtype=torch.int32)
        lengths_5 = None
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
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.conv2d(
            input_2,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
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
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        input_6 = torch.conv2d(
            input_5,
            l_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        input_5 = (
            l_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_ = None
        input_7 = torch.conv2d(
            input_6,
            l_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_6 = (
            l_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_ = None
        input_8 = torch.nn.functional.relu(input_7, inplace=True)
        input_7 = None
        transpose_1 = input_8.transpose(1, 2)
        input_8 = None
        reshape = transpose_1.reshape(1, 66, -1)
        transpose_1 = None
        x_1 = torch._C._nn.linear(
            reshape,
            l_instance_modules_pre_encode_modules_out_parameters_weight_,
            l_instance_modules_pre_encode_modules_out_parameters_bias_,
        )
        reshape = (
            l_instance_modules_pre_encode_modules_out_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_out_parameters_bias_ = None
        length = lengths_6.to(torch.int64)
        lengths_6 = None
        pos_emb = l_instance_modules_pos_enc_buffers_pe_[
            (slice(None, None, None), slice(4934, 5065, None))
        ]
        l_instance_modules_pos_enc_buffers_pe_ = None
        audio_signal_1 = torch.nn.functional.dropout(x_1, 0.1, False, False)
        x_1 = None
        att_mask = torch.ones(
            1, 66, 66, dtype=torch.bool, device=device(type="cuda", index=0)
        )
        arange = torch.arange(0, 66, device=device(type="cuda", index=0))
        expand = arange.expand(1, -1)
        arange = None
        unsqueeze_1 = length.unsqueeze(-1)
        pad_mask = expand < unsqueeze_1
        expand = unsqueeze_1 = None
        unsqueeze_2 = pad_mask.unsqueeze(1)
        pad_mask_for_att_mask = unsqueeze_2.repeat([1, 66, 1])
        unsqueeze_2 = None
        transpose_2 = pad_mask_for_att_mask.transpose(1, 2)
        pad_mask_for_att_mask_1 = torch.logical_and(pad_mask_for_att_mask, transpose_2)
        pad_mask_for_att_mask = transpose_2 = None
        att_mask_1 = att_mask[
            (slice(None, None, None), slice(None, 66, None), slice(None, 66, None))
        ]
        att_mask = None
        to_5 = att_mask_1.to(device(type="cuda", index=0))
        att_mask_1 = None
        att_mask_2 = torch.logical_and(pad_mask_for_att_mask_1, to_5)
        pad_mask_for_att_mask_1 = to_5 = None
        att_mask_3 = ~att_mask_2
        att_mask_2 = None
        pad_mask_1 = ~pad_mask
        pad_mask = None
        x_2 = torch.nn.functional.layer_norm(
            audio_signal_1,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_3 = torch._C._nn.linear(
            x_2,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_2 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_4 = torch.nn.functional.silu(x_3, inplace=False)
        x_3 = None
        x_5 = torch.nn.functional.dropout(x_4, 0.1, False, False)
        x_4 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_5 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_2 = torch.nn.functional.dropout(x_6, 0.1, False, False)
        x_6 = None
        mul = dropout_2 * 0.5
        dropout_2 = None
        residual = audio_signal_1 + mul
        audio_signal_1 = mul = None
        x_7 = torch.nn.functional.layer_norm(
            residual,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_
        ) = None
        linear_3 = torch._C._nn.linear(
            x_7,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q = linear_3.view(1, -1, 8, 64)
        linear_3 = None
        linear_4 = torch._C._nn.linear(
            x_7,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k = linear_4.view(1, -1, 8, 64)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            x_7,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_7 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v = linear_5.view(1, -1, 8, 64)
        linear_5 = None
        q_1 = q.transpose(1, 2)
        q = None
        k_1 = k.transpose(1, 2)
        k = None
        v_1 = v.transpose(1, 2)
        v = None
        q_2 = q_1.transpose(1, 2)
        q_1 = None
        linear_6 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p = linear_6.view(1, -1, 8, 64)
        linear_6 = None
        p_1 = p.transpose(1, 2)
        p = None
        add_7 = (
            q_2
            + l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u = add_7.transpose(1, 2)
        add_7 = None
        add_8 = (
            q_2
            + l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_
        )
        q_2 = (
            l_instance_modules_layers_modules_0_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v = add_8.transpose(1, 2)
        add_8 = None
        transpose_10 = p_1.transpose(-2, -1)
        p_1 = None
        matrix_bd = torch.matmul(q_with_bias_v, transpose_10)
        q_with_bias_v = transpose_10 = None
        x_8 = torch._C._nn.pad(matrix_bd, (1, 0), "constant", None)
        matrix_bd = None
        x_9 = x_8.view(1, 8, -1, 66)
        x_8 = None
        getitem_2 = x_9[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_9 = None
        x_10 = getitem_2.view(1, 8, 66, 131)
        getitem_2 = None
        transpose_11 = k_1.transpose(-2, -1)
        k_1 = None
        matrix_ac = torch.matmul(q_with_bias_u, transpose_11)
        q_with_bias_u = transpose_11 = None
        matrix_bd_1 = x_10[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_10 = None
        add_9 = matrix_ac + matrix_bd_1
        matrix_ac = matrix_bd_1 = None
        scores = add_9 / 8.0
        add_9 = None
        mask = att_mask_3.unsqueeze(1)
        scores_1 = scores.masked_fill(mask, -10000.0)
        scores = None
        softmax = torch.softmax(scores_1, dim=-1)
        scores_1 = None
        attn = softmax.masked_fill(mask, 0.0)
        softmax = mask = None
        p_attn = torch.nn.functional.dropout(attn, 0.1, False, False)
        attn = None
        x_11 = torch.matmul(p_attn, v_1)
        p_attn = v_1 = None
        transpose_12 = x_11.transpose(1, 2)
        x_11 = None
        x_12 = transpose_12.reshape(1, -1, 512)
        transpose_12 = None
        out = torch._C._nn.linear(
            x_12,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_12 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_4 = torch.nn.functional.dropout(out, 0.1, False, False)
        out = None
        residual_1 = residual + dropout_4
        residual = dropout_4 = None
        x_13 = torch.nn.functional.layer_norm(
            residual_1,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_
        ) = None
        x_14 = x_13.transpose(1, 2)
        x_13 = None
        x_15 = torch.conv1d(
            x_14,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_14 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_16 = torch.nn.functional.glu(x_15, dim=1)
        x_15 = None
        unsqueeze_4 = pad_mask_1.unsqueeze(1)
        x_17 = x_16.masked_fill(unsqueeze_4, 0.0)
        x_16 = unsqueeze_4 = None
        new_x = torch._C._nn.pad(x_17, (4, 4), "constant", None)
        x_17 = None
        x_18 = torch.conv1d(
            new_x,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_20 = torch.nn.functional.silu(x_19, inplace=False)
        x_19 = None
        x_21 = torch.conv1d(
            x_20,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_20 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_22 = x_21.transpose(1, 2)
        x_21 = None
        dropout_5 = torch.nn.functional.dropout(x_22, 0.1, False, False)
        x_22 = None
        residual_2 = residual_1 + dropout_5
        residual_1 = dropout_5 = None
        x_23 = torch.nn.functional.layer_norm(
            residual_2,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_24 = torch._C._nn.linear(
            x_23,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_23 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_25 = torch.nn.functional.silu(x_24, inplace=False)
        x_24 = None
        x_26 = torch.nn.functional.dropout(x_25, 0.1, False, False)
        x_25 = None
        x_27 = torch._C._nn.linear(
            x_26,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_26 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_7 = torch.nn.functional.dropout(x_27, 0.1, False, False)
        x_27 = None
        mul_1 = dropout_7 * 0.5
        dropout_7 = None
        residual_3 = residual_2 + mul_1
        residual_2 = mul_1 = None
        x_28 = torch.nn.functional.layer_norm(
            residual_3,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_3 = (
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_ = None
        x_29 = torch.nn.functional.layer_norm(
            x_28,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_30 = torch._C._nn.linear(
            x_29,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_29 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_31 = torch.nn.functional.silu(x_30, inplace=False)
        x_30 = None
        x_32 = torch.nn.functional.dropout(x_31, 0.1, False, False)
        x_31 = None
        x_33 = torch._C._nn.linear(
            x_32,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_32 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_9 = torch.nn.functional.dropout(x_33, 0.1, False, False)
        x_33 = None
        mul_2 = dropout_9 * 0.5
        dropout_9 = None
        residual_4 = x_28 + mul_2
        x_28 = mul_2 = None
        x_34 = torch.nn.functional.layer_norm(
            residual_4,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_34,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_3 = linear_12.view(1, -1, 8, 64)
        linear_12 = None
        linear_13 = torch._C._nn.linear(
            x_34,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_2 = linear_13.view(1, -1, 8, 64)
        linear_13 = None
        linear_14 = torch._C._nn.linear(
            x_34,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_34 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_2 = linear_14.view(1, -1, 8, 64)
        linear_14 = None
        q_4 = q_3.transpose(1, 2)
        q_3 = None
        k_3 = k_2.transpose(1, 2)
        k_2 = None
        v_3 = v_2.transpose(1, 2)
        v_2 = None
        q_5 = q_4.transpose(1, 2)
        q_4 = None
        linear_15 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_2 = linear_15.view(1, -1, 8, 64)
        linear_15 = None
        p_3 = p_2.transpose(1, 2)
        p_2 = None
        add_14 = (
            q_5
            + l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_1 = add_14.transpose(1, 2)
        add_14 = None
        add_15 = (
            q_5
            + l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_
        )
        q_5 = (
            l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_1 = add_15.transpose(1, 2)
        add_15 = None
        transpose_22 = p_3.transpose(-2, -1)
        p_3 = None
        matrix_bd_2 = torch.matmul(q_with_bias_v_1, transpose_22)
        q_with_bias_v_1 = transpose_22 = None
        x_35 = torch._C._nn.pad(matrix_bd_2, (1, 0), "constant", None)
        matrix_bd_2 = None
        x_36 = x_35.view(1, 8, -1, 66)
        x_35 = None
        getitem_4 = x_36[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_36 = None
        x_37 = getitem_4.view(1, 8, 66, 131)
        getitem_4 = None
        transpose_23 = k_3.transpose(-2, -1)
        k_3 = None
        matrix_ac_1 = torch.matmul(q_with_bias_u_1, transpose_23)
        q_with_bias_u_1 = transpose_23 = None
        matrix_bd_3 = x_37[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_37 = None
        add_16 = matrix_ac_1 + matrix_bd_3
        matrix_ac_1 = matrix_bd_3 = None
        scores_2 = add_16 / 8.0
        add_16 = None
        mask_1 = att_mask_3.unsqueeze(1)
        scores_3 = scores_2.masked_fill(mask_1, -10000.0)
        scores_2 = None
        softmax_1 = torch.softmax(scores_3, dim=-1)
        scores_3 = None
        attn_1 = softmax_1.masked_fill(mask_1, 0.0)
        softmax_1 = mask_1 = None
        p_attn_1 = torch.nn.functional.dropout(attn_1, 0.1, False, False)
        attn_1 = None
        x_38 = torch.matmul(p_attn_1, v_3)
        p_attn_1 = v_3 = None
        transpose_24 = x_38.transpose(1, 2)
        x_38 = None
        x_39 = transpose_24.reshape(1, -1, 512)
        transpose_24 = None
        out_1 = torch._C._nn.linear(
            x_39,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_39 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_11 = torch.nn.functional.dropout(out_1, 0.1, False, False)
        out_1 = None
        residual_5 = residual_4 + dropout_11
        residual_4 = dropout_11 = None
        x_40 = torch.nn.functional.layer_norm(
            residual_5,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_
        ) = None
        x_41 = x_40.transpose(1, 2)
        x_40 = None
        x_42 = torch.conv1d(
            x_41,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_41 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_43 = torch.nn.functional.glu(x_42, dim=1)
        x_42 = None
        unsqueeze_6 = pad_mask_1.unsqueeze(1)
        x_44 = x_43.masked_fill(unsqueeze_6, 0.0)
        x_43 = unsqueeze_6 = None
        new_x_1 = torch._C._nn.pad(x_44, (4, 4), "constant", None)
        x_44 = None
        x_45 = torch.conv1d(
            new_x_1,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_1 = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_47 = torch.nn.functional.silu(x_46, inplace=False)
        x_46 = None
        x_48 = torch.conv1d(
            x_47,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_47 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_49 = x_48.transpose(1, 2)
        x_48 = None
        dropout_12 = torch.nn.functional.dropout(x_49, 0.1, False, False)
        x_49 = None
        residual_6 = residual_5 + dropout_12
        residual_5 = dropout_12 = None
        x_50 = torch.nn.functional.layer_norm(
            residual_6,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_51 = torch._C._nn.linear(
            x_50,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_50 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_52 = torch.nn.functional.silu(x_51, inplace=False)
        x_51 = None
        x_53 = torch.nn.functional.dropout(x_52, 0.1, False, False)
        x_52 = None
        x_54 = torch._C._nn.linear(
            x_53,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_53 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_14 = torch.nn.functional.dropout(x_54, 0.1, False, False)
        x_54 = None
        mul_3 = dropout_14 * 0.5
        dropout_14 = None
        residual_7 = residual_6 + mul_3
        residual_6 = mul_3 = None
        x_55 = torch.nn.functional.layer_norm(
            residual_7,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_7 = (
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_ = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_57 = torch._C._nn.linear(
            x_56,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_56 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_58 = torch.nn.functional.silu(x_57, inplace=False)
        x_57 = None
        x_59 = torch.nn.functional.dropout(x_58, 0.1, False, False)
        x_58 = None
        x_60 = torch._C._nn.linear(
            x_59,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_59 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_16 = torch.nn.functional.dropout(x_60, 0.1, False, False)
        x_60 = None
        mul_4 = dropout_16 * 0.5
        dropout_16 = None
        residual_8 = x_55 + mul_4
        x_55 = mul_4 = None
        x_61 = torch.nn.functional.layer_norm(
            residual_8,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_
        ) = None
        linear_21 = torch._C._nn.linear(
            x_61,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_6 = linear_21.view(1, -1, 8, 64)
        linear_21 = None
        linear_22 = torch._C._nn.linear(
            x_61,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_4 = linear_22.view(1, -1, 8, 64)
        linear_22 = None
        linear_23 = torch._C._nn.linear(
            x_61,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_61 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_4 = linear_23.view(1, -1, 8, 64)
        linear_23 = None
        q_7 = q_6.transpose(1, 2)
        q_6 = None
        k_5 = k_4.transpose(1, 2)
        k_4 = None
        v_5 = v_4.transpose(1, 2)
        v_4 = None
        q_8 = q_7.transpose(1, 2)
        q_7 = None
        linear_24 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_4 = linear_24.view(1, -1, 8, 64)
        linear_24 = None
        p_5 = p_4.transpose(1, 2)
        p_4 = None
        add_21 = (
            q_8
            + l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_2 = add_21.transpose(1, 2)
        add_21 = None
        add_22 = (
            q_8
            + l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_
        )
        q_8 = (
            l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_2 = add_22.transpose(1, 2)
        add_22 = None
        transpose_34 = p_5.transpose(-2, -1)
        p_5 = None
        matrix_bd_4 = torch.matmul(q_with_bias_v_2, transpose_34)
        q_with_bias_v_2 = transpose_34 = None
        x_62 = torch._C._nn.pad(matrix_bd_4, (1, 0), "constant", None)
        matrix_bd_4 = None
        x_63 = x_62.view(1, 8, -1, 66)
        x_62 = None
        getitem_6 = x_63[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_63 = None
        x_64 = getitem_6.view(1, 8, 66, 131)
        getitem_6 = None
        transpose_35 = k_5.transpose(-2, -1)
        k_5 = None
        matrix_ac_2 = torch.matmul(q_with_bias_u_2, transpose_35)
        q_with_bias_u_2 = transpose_35 = None
        matrix_bd_5 = x_64[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_64 = None
        add_23 = matrix_ac_2 + matrix_bd_5
        matrix_ac_2 = matrix_bd_5 = None
        scores_4 = add_23 / 8.0
        add_23 = None
        mask_2 = att_mask_3.unsqueeze(1)
        scores_5 = scores_4.masked_fill(mask_2, -10000.0)
        scores_4 = None
        softmax_2 = torch.softmax(scores_5, dim=-1)
        scores_5 = None
        attn_2 = softmax_2.masked_fill(mask_2, 0.0)
        softmax_2 = mask_2 = None
        p_attn_2 = torch.nn.functional.dropout(attn_2, 0.1, False, False)
        attn_2 = None
        x_65 = torch.matmul(p_attn_2, v_5)
        p_attn_2 = v_5 = None
        transpose_36 = x_65.transpose(1, 2)
        x_65 = None
        x_66 = transpose_36.reshape(1, -1, 512)
        transpose_36 = None
        out_2 = torch._C._nn.linear(
            x_66,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_66 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_18 = torch.nn.functional.dropout(out_2, 0.1, False, False)
        out_2 = None
        residual_9 = residual_8 + dropout_18
        residual_8 = dropout_18 = None
        x_67 = torch.nn.functional.layer_norm(
            residual_9,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_
        ) = None
        x_68 = x_67.transpose(1, 2)
        x_67 = None
        x_69 = torch.conv1d(
            x_68,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_68 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_70 = torch.nn.functional.glu(x_69, dim=1)
        x_69 = None
        unsqueeze_8 = pad_mask_1.unsqueeze(1)
        x_71 = x_70.masked_fill(unsqueeze_8, 0.0)
        x_70 = unsqueeze_8 = None
        new_x_2 = torch._C._nn.pad(x_71, (4, 4), "constant", None)
        x_71 = None
        x_72 = torch.conv1d(
            new_x_2,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_2 = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_74 = torch.nn.functional.silu(x_73, inplace=False)
        x_73 = None
        x_75 = torch.conv1d(
            x_74,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_74 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_76 = x_75.transpose(1, 2)
        x_75 = None
        dropout_19 = torch.nn.functional.dropout(x_76, 0.1, False, False)
        x_76 = None
        residual_10 = residual_9 + dropout_19
        residual_9 = dropout_19 = None
        x_77 = torch.nn.functional.layer_norm(
            residual_10,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_78 = torch._C._nn.linear(
            x_77,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_77 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_79 = torch.nn.functional.silu(x_78, inplace=False)
        x_78 = None
        x_80 = torch.nn.functional.dropout(x_79, 0.1, False, False)
        x_79 = None
        x_81 = torch._C._nn.linear(
            x_80,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_80 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_21 = torch.nn.functional.dropout(x_81, 0.1, False, False)
        x_81 = None
        mul_5 = dropout_21 * 0.5
        dropout_21 = None
        residual_11 = residual_10 + mul_5
        residual_10 = mul_5 = None
        x_82 = torch.nn.functional.layer_norm(
            residual_11,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_11 = (
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_ = None
        x_83 = torch.nn.functional.layer_norm(
            x_82,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_84 = torch._C._nn.linear(
            x_83,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_83 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_85 = torch.nn.functional.silu(x_84, inplace=False)
        x_84 = None
        x_86 = torch.nn.functional.dropout(x_85, 0.1, False, False)
        x_85 = None
        x_87 = torch._C._nn.linear(
            x_86,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_86 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_23 = torch.nn.functional.dropout(x_87, 0.1, False, False)
        x_87 = None
        mul_6 = dropout_23 * 0.5
        dropout_23 = None
        residual_12 = x_82 + mul_6
        x_82 = mul_6 = None
        x_88 = torch.nn.functional.layer_norm(
            residual_12,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_
        ) = None
        linear_30 = torch._C._nn.linear(
            x_88,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_9 = linear_30.view(1, -1, 8, 64)
        linear_30 = None
        linear_31 = torch._C._nn.linear(
            x_88,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_6 = linear_31.view(1, -1, 8, 64)
        linear_31 = None
        linear_32 = torch._C._nn.linear(
            x_88,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_88 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_6 = linear_32.view(1, -1, 8, 64)
        linear_32 = None
        q_10 = q_9.transpose(1, 2)
        q_9 = None
        k_7 = k_6.transpose(1, 2)
        k_6 = None
        v_7 = v_6.transpose(1, 2)
        v_6 = None
        q_11 = q_10.transpose(1, 2)
        q_10 = None
        linear_33 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_6 = linear_33.view(1, -1, 8, 64)
        linear_33 = None
        p_7 = p_6.transpose(1, 2)
        p_6 = None
        add_28 = (
            q_11
            + l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_3 = add_28.transpose(1, 2)
        add_28 = None
        add_29 = (
            q_11
            + l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_
        )
        q_11 = (
            l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_3 = add_29.transpose(1, 2)
        add_29 = None
        transpose_46 = p_7.transpose(-2, -1)
        p_7 = None
        matrix_bd_6 = torch.matmul(q_with_bias_v_3, transpose_46)
        q_with_bias_v_3 = transpose_46 = None
        x_89 = torch._C._nn.pad(matrix_bd_6, (1, 0), "constant", None)
        matrix_bd_6 = None
        x_90 = x_89.view(1, 8, -1, 66)
        x_89 = None
        getitem_8 = x_90[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_90 = None
        x_91 = getitem_8.view(1, 8, 66, 131)
        getitem_8 = None
        transpose_47 = k_7.transpose(-2, -1)
        k_7 = None
        matrix_ac_3 = torch.matmul(q_with_bias_u_3, transpose_47)
        q_with_bias_u_3 = transpose_47 = None
        matrix_bd_7 = x_91[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_91 = None
        add_30 = matrix_ac_3 + matrix_bd_7
        matrix_ac_3 = matrix_bd_7 = None
        scores_6 = add_30 / 8.0
        add_30 = None
        mask_3 = att_mask_3.unsqueeze(1)
        scores_7 = scores_6.masked_fill(mask_3, -10000.0)
        scores_6 = None
        softmax_3 = torch.softmax(scores_7, dim=-1)
        scores_7 = None
        attn_3 = softmax_3.masked_fill(mask_3, 0.0)
        softmax_3 = mask_3 = None
        p_attn_3 = torch.nn.functional.dropout(attn_3, 0.1, False, False)
        attn_3 = None
        x_92 = torch.matmul(p_attn_3, v_7)
        p_attn_3 = v_7 = None
        transpose_48 = x_92.transpose(1, 2)
        x_92 = None
        x_93 = transpose_48.reshape(1, -1, 512)
        transpose_48 = None
        out_3 = torch._C._nn.linear(
            x_93,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_93 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_25 = torch.nn.functional.dropout(out_3, 0.1, False, False)
        out_3 = None
        residual_13 = residual_12 + dropout_25
        residual_12 = dropout_25 = None
        x_94 = torch.nn.functional.layer_norm(
            residual_13,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_
        ) = None
        x_95 = x_94.transpose(1, 2)
        x_94 = None
        x_96 = torch.conv1d(
            x_95,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_95 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_97 = torch.nn.functional.glu(x_96, dim=1)
        x_96 = None
        unsqueeze_10 = pad_mask_1.unsqueeze(1)
        x_98 = x_97.masked_fill(unsqueeze_10, 0.0)
        x_97 = unsqueeze_10 = None
        new_x_3 = torch._C._nn.pad(x_98, (4, 4), "constant", None)
        x_98 = None
        x_99 = torch.conv1d(
            new_x_3,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_3 = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_101 = torch.nn.functional.silu(x_100, inplace=False)
        x_100 = None
        x_102 = torch.conv1d(
            x_101,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_101 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_103 = x_102.transpose(1, 2)
        x_102 = None
        dropout_26 = torch.nn.functional.dropout(x_103, 0.1, False, False)
        x_103 = None
        residual_14 = residual_13 + dropout_26
        residual_13 = dropout_26 = None
        x_104 = torch.nn.functional.layer_norm(
            residual_14,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_105 = torch._C._nn.linear(
            x_104,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_104 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_106 = torch.nn.functional.silu(x_105, inplace=False)
        x_105 = None
        x_107 = torch.nn.functional.dropout(x_106, 0.1, False, False)
        x_106 = None
        x_108 = torch._C._nn.linear(
            x_107,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_107 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_28 = torch.nn.functional.dropout(x_108, 0.1, False, False)
        x_108 = None
        mul_7 = dropout_28 * 0.5
        dropout_28 = None
        residual_15 = residual_14 + mul_7
        residual_14 = mul_7 = None
        x_109 = torch.nn.functional.layer_norm(
            residual_15,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_15 = (
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_ = None
        x_110 = torch.nn.functional.layer_norm(
            x_109,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_111 = torch._C._nn.linear(
            x_110,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_110 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_112 = torch.nn.functional.silu(x_111, inplace=False)
        x_111 = None
        x_113 = torch.nn.functional.dropout(x_112, 0.1, False, False)
        x_112 = None
        x_114 = torch._C._nn.linear(
            x_113,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_113 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_30 = torch.nn.functional.dropout(x_114, 0.1, False, False)
        x_114 = None
        mul_8 = dropout_30 * 0.5
        dropout_30 = None
        residual_16 = x_109 + mul_8
        x_109 = mul_8 = None
        x_115 = torch.nn.functional.layer_norm(
            residual_16,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_
        ) = None
        linear_39 = torch._C._nn.linear(
            x_115,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_12 = linear_39.view(1, -1, 8, 64)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            x_115,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_8 = linear_40.view(1, -1, 8, 64)
        linear_40 = None
        linear_41 = torch._C._nn.linear(
            x_115,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_115 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_8 = linear_41.view(1, -1, 8, 64)
        linear_41 = None
        q_13 = q_12.transpose(1, 2)
        q_12 = None
        k_9 = k_8.transpose(1, 2)
        k_8 = None
        v_9 = v_8.transpose(1, 2)
        v_8 = None
        q_14 = q_13.transpose(1, 2)
        q_13 = None
        linear_42 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_8 = linear_42.view(1, -1, 8, 64)
        linear_42 = None
        p_9 = p_8.transpose(1, 2)
        p_8 = None
        add_35 = (
            q_14
            + l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_4 = add_35.transpose(1, 2)
        add_35 = None
        add_36 = (
            q_14
            + l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_
        )
        q_14 = (
            l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_4 = add_36.transpose(1, 2)
        add_36 = None
        transpose_58 = p_9.transpose(-2, -1)
        p_9 = None
        matrix_bd_8 = torch.matmul(q_with_bias_v_4, transpose_58)
        q_with_bias_v_4 = transpose_58 = None
        x_116 = torch._C._nn.pad(matrix_bd_8, (1, 0), "constant", None)
        matrix_bd_8 = None
        x_117 = x_116.view(1, 8, -1, 66)
        x_116 = None
        getitem_10 = x_117[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_117 = None
        x_118 = getitem_10.view(1, 8, 66, 131)
        getitem_10 = None
        transpose_59 = k_9.transpose(-2, -1)
        k_9 = None
        matrix_ac_4 = torch.matmul(q_with_bias_u_4, transpose_59)
        q_with_bias_u_4 = transpose_59 = None
        matrix_bd_9 = x_118[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_118 = None
        add_37 = matrix_ac_4 + matrix_bd_9
        matrix_ac_4 = matrix_bd_9 = None
        scores_8 = add_37 / 8.0
        add_37 = None
        mask_4 = att_mask_3.unsqueeze(1)
        scores_9 = scores_8.masked_fill(mask_4, -10000.0)
        scores_8 = None
        softmax_4 = torch.softmax(scores_9, dim=-1)
        scores_9 = None
        attn_4 = softmax_4.masked_fill(mask_4, 0.0)
        softmax_4 = mask_4 = None
        p_attn_4 = torch.nn.functional.dropout(attn_4, 0.1, False, False)
        attn_4 = None
        x_119 = torch.matmul(p_attn_4, v_9)
        p_attn_4 = v_9 = None
        transpose_60 = x_119.transpose(1, 2)
        x_119 = None
        x_120 = transpose_60.reshape(1, -1, 512)
        transpose_60 = None
        out_4 = torch._C._nn.linear(
            x_120,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_120 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_32 = torch.nn.functional.dropout(out_4, 0.1, False, False)
        out_4 = None
        residual_17 = residual_16 + dropout_32
        residual_16 = dropout_32 = None
        x_121 = torch.nn.functional.layer_norm(
            residual_17,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_
        ) = None
        x_122 = x_121.transpose(1, 2)
        x_121 = None
        x_123 = torch.conv1d(
            x_122,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_122 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_124 = torch.nn.functional.glu(x_123, dim=1)
        x_123 = None
        unsqueeze_12 = pad_mask_1.unsqueeze(1)
        x_125 = x_124.masked_fill(unsqueeze_12, 0.0)
        x_124 = unsqueeze_12 = None
        new_x_4 = torch._C._nn.pad(x_125, (4, 4), "constant", None)
        x_125 = None
        x_126 = torch.conv1d(
            new_x_4,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_4 = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_128 = torch.nn.functional.silu(x_127, inplace=False)
        x_127 = None
        x_129 = torch.conv1d(
            x_128,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_128 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_130 = x_129.transpose(1, 2)
        x_129 = None
        dropout_33 = torch.nn.functional.dropout(x_130, 0.1, False, False)
        x_130 = None
        residual_18 = residual_17 + dropout_33
        residual_17 = dropout_33 = None
        x_131 = torch.nn.functional.layer_norm(
            residual_18,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_132 = torch._C._nn.linear(
            x_131,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_131 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_133 = torch.nn.functional.silu(x_132, inplace=False)
        x_132 = None
        x_134 = torch.nn.functional.dropout(x_133, 0.1, False, False)
        x_133 = None
        x_135 = torch._C._nn.linear(
            x_134,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_134 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_35 = torch.nn.functional.dropout(x_135, 0.1, False, False)
        x_135 = None
        mul_9 = dropout_35 * 0.5
        dropout_35 = None
        residual_19 = residual_18 + mul_9
        residual_18 = mul_9 = None
        x_136 = torch.nn.functional.layer_norm(
            residual_19,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_19 = (
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_ = None
        x_137 = torch.nn.functional.layer_norm(
            x_136,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_138 = torch._C._nn.linear(
            x_137,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_137 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_139 = torch.nn.functional.silu(x_138, inplace=False)
        x_138 = None
        x_140 = torch.nn.functional.dropout(x_139, 0.1, False, False)
        x_139 = None
        x_141 = torch._C._nn.linear(
            x_140,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_140 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_37 = torch.nn.functional.dropout(x_141, 0.1, False, False)
        x_141 = None
        mul_10 = dropout_37 * 0.5
        dropout_37 = None
        residual_20 = x_136 + mul_10
        x_136 = mul_10 = None
        x_142 = torch.nn.functional.layer_norm(
            residual_20,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            x_142,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_15 = linear_48.view(1, -1, 8, 64)
        linear_48 = None
        linear_49 = torch._C._nn.linear(
            x_142,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_10 = linear_49.view(1, -1, 8, 64)
        linear_49 = None
        linear_50 = torch._C._nn.linear(
            x_142,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_142 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_10 = linear_50.view(1, -1, 8, 64)
        linear_50 = None
        q_16 = q_15.transpose(1, 2)
        q_15 = None
        k_11 = k_10.transpose(1, 2)
        k_10 = None
        v_11 = v_10.transpose(1, 2)
        v_10 = None
        q_17 = q_16.transpose(1, 2)
        q_16 = None
        linear_51 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_10 = linear_51.view(1, -1, 8, 64)
        linear_51 = None
        p_11 = p_10.transpose(1, 2)
        p_10 = None
        add_42 = (
            q_17
            + l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_5 = add_42.transpose(1, 2)
        add_42 = None
        add_43 = (
            q_17
            + l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_
        )
        q_17 = (
            l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_5 = add_43.transpose(1, 2)
        add_43 = None
        transpose_70 = p_11.transpose(-2, -1)
        p_11 = None
        matrix_bd_10 = torch.matmul(q_with_bias_v_5, transpose_70)
        q_with_bias_v_5 = transpose_70 = None
        x_143 = torch._C._nn.pad(matrix_bd_10, (1, 0), "constant", None)
        matrix_bd_10 = None
        x_144 = x_143.view(1, 8, -1, 66)
        x_143 = None
        getitem_12 = x_144[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_144 = None
        x_145 = getitem_12.view(1, 8, 66, 131)
        getitem_12 = None
        transpose_71 = k_11.transpose(-2, -1)
        k_11 = None
        matrix_ac_5 = torch.matmul(q_with_bias_u_5, transpose_71)
        q_with_bias_u_5 = transpose_71 = None
        matrix_bd_11 = x_145[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_145 = None
        add_44 = matrix_ac_5 + matrix_bd_11
        matrix_ac_5 = matrix_bd_11 = None
        scores_10 = add_44 / 8.0
        add_44 = None
        mask_5 = att_mask_3.unsqueeze(1)
        scores_11 = scores_10.masked_fill(mask_5, -10000.0)
        scores_10 = None
        softmax_5 = torch.softmax(scores_11, dim=-1)
        scores_11 = None
        attn_5 = softmax_5.masked_fill(mask_5, 0.0)
        softmax_5 = mask_5 = None
        p_attn_5 = torch.nn.functional.dropout(attn_5, 0.1, False, False)
        attn_5 = None
        x_146 = torch.matmul(p_attn_5, v_11)
        p_attn_5 = v_11 = None
        transpose_72 = x_146.transpose(1, 2)
        x_146 = None
        x_147 = transpose_72.reshape(1, -1, 512)
        transpose_72 = None
        out_5 = torch._C._nn.linear(
            x_147,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_147 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_39 = torch.nn.functional.dropout(out_5, 0.1, False, False)
        out_5 = None
        residual_21 = residual_20 + dropout_39
        residual_20 = dropout_39 = None
        x_148 = torch.nn.functional.layer_norm(
            residual_21,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_
        ) = None
        x_149 = x_148.transpose(1, 2)
        x_148 = None
        x_150 = torch.conv1d(
            x_149,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_149 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_151 = torch.nn.functional.glu(x_150, dim=1)
        x_150 = None
        unsqueeze_14 = pad_mask_1.unsqueeze(1)
        x_152 = x_151.masked_fill(unsqueeze_14, 0.0)
        x_151 = unsqueeze_14 = None
        new_x_5 = torch._C._nn.pad(x_152, (4, 4), "constant", None)
        x_152 = None
        x_153 = torch.conv1d(
            new_x_5,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_5 = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_155 = torch.nn.functional.silu(x_154, inplace=False)
        x_154 = None
        x_156 = torch.conv1d(
            x_155,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_155 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_157 = x_156.transpose(1, 2)
        x_156 = None
        dropout_40 = torch.nn.functional.dropout(x_157, 0.1, False, False)
        x_157 = None
        residual_22 = residual_21 + dropout_40
        residual_21 = dropout_40 = None
        x_158 = torch.nn.functional.layer_norm(
            residual_22,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_159 = torch._C._nn.linear(
            x_158,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_158 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_160 = torch.nn.functional.silu(x_159, inplace=False)
        x_159 = None
        x_161 = torch.nn.functional.dropout(x_160, 0.1, False, False)
        x_160 = None
        x_162 = torch._C._nn.linear(
            x_161,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_161 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_42 = torch.nn.functional.dropout(x_162, 0.1, False, False)
        x_162 = None
        mul_11 = dropout_42 * 0.5
        dropout_42 = None
        residual_23 = residual_22 + mul_11
        residual_22 = mul_11 = None
        x_163 = torch.nn.functional.layer_norm(
            residual_23,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_23 = (
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_ = None
        x_164 = torch.nn.functional.layer_norm(
            x_163,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_165 = torch._C._nn.linear(
            x_164,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_164 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_166 = torch.nn.functional.silu(x_165, inplace=False)
        x_165 = None
        x_167 = torch.nn.functional.dropout(x_166, 0.1, False, False)
        x_166 = None
        x_168 = torch._C._nn.linear(
            x_167,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_167 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_44 = torch.nn.functional.dropout(x_168, 0.1, False, False)
        x_168 = None
        mul_12 = dropout_44 * 0.5
        dropout_44 = None
        residual_24 = x_163 + mul_12
        x_163 = mul_12 = None
        x_169 = torch.nn.functional.layer_norm(
            residual_24,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_
        ) = None
        linear_57 = torch._C._nn.linear(
            x_169,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_18 = linear_57.view(1, -1, 8, 64)
        linear_57 = None
        linear_58 = torch._C._nn.linear(
            x_169,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_12 = linear_58.view(1, -1, 8, 64)
        linear_58 = None
        linear_59 = torch._C._nn.linear(
            x_169,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_169 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_12 = linear_59.view(1, -1, 8, 64)
        linear_59 = None
        q_19 = q_18.transpose(1, 2)
        q_18 = None
        k_13 = k_12.transpose(1, 2)
        k_12 = None
        v_13 = v_12.transpose(1, 2)
        v_12 = None
        q_20 = q_19.transpose(1, 2)
        q_19 = None
        linear_60 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_12 = linear_60.view(1, -1, 8, 64)
        linear_60 = None
        p_13 = p_12.transpose(1, 2)
        p_12 = None
        add_49 = (
            q_20
            + l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_6 = add_49.transpose(1, 2)
        add_49 = None
        add_50 = (
            q_20
            + l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_
        )
        q_20 = (
            l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_6 = add_50.transpose(1, 2)
        add_50 = None
        transpose_82 = p_13.transpose(-2, -1)
        p_13 = None
        matrix_bd_12 = torch.matmul(q_with_bias_v_6, transpose_82)
        q_with_bias_v_6 = transpose_82 = None
        x_170 = torch._C._nn.pad(matrix_bd_12, (1, 0), "constant", None)
        matrix_bd_12 = None
        x_171 = x_170.view(1, 8, -1, 66)
        x_170 = None
        getitem_14 = x_171[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_171 = None
        x_172 = getitem_14.view(1, 8, 66, 131)
        getitem_14 = None
        transpose_83 = k_13.transpose(-2, -1)
        k_13 = None
        matrix_ac_6 = torch.matmul(q_with_bias_u_6, transpose_83)
        q_with_bias_u_6 = transpose_83 = None
        matrix_bd_13 = x_172[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_172 = None
        add_51 = matrix_ac_6 + matrix_bd_13
        matrix_ac_6 = matrix_bd_13 = None
        scores_12 = add_51 / 8.0
        add_51 = None
        mask_6 = att_mask_3.unsqueeze(1)
        scores_13 = scores_12.masked_fill(mask_6, -10000.0)
        scores_12 = None
        softmax_6 = torch.softmax(scores_13, dim=-1)
        scores_13 = None
        attn_6 = softmax_6.masked_fill(mask_6, 0.0)
        softmax_6 = mask_6 = None
        p_attn_6 = torch.nn.functional.dropout(attn_6, 0.1, False, False)
        attn_6 = None
        x_173 = torch.matmul(p_attn_6, v_13)
        p_attn_6 = v_13 = None
        transpose_84 = x_173.transpose(1, 2)
        x_173 = None
        x_174 = transpose_84.reshape(1, -1, 512)
        transpose_84 = None
        out_6 = torch._C._nn.linear(
            x_174,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_174 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_46 = torch.nn.functional.dropout(out_6, 0.1, False, False)
        out_6 = None
        residual_25 = residual_24 + dropout_46
        residual_24 = dropout_46 = None
        x_175 = torch.nn.functional.layer_norm(
            residual_25,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_
        ) = None
        x_176 = x_175.transpose(1, 2)
        x_175 = None
        x_177 = torch.conv1d(
            x_176,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_176 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_178 = torch.nn.functional.glu(x_177, dim=1)
        x_177 = None
        unsqueeze_16 = pad_mask_1.unsqueeze(1)
        x_179 = x_178.masked_fill(unsqueeze_16, 0.0)
        x_178 = unsqueeze_16 = None
        new_x_6 = torch._C._nn.pad(x_179, (4, 4), "constant", None)
        x_179 = None
        x_180 = torch.conv1d(
            new_x_6,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_6 = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_182 = torch.nn.functional.silu(x_181, inplace=False)
        x_181 = None
        x_183 = torch.conv1d(
            x_182,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_182 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_184 = x_183.transpose(1, 2)
        x_183 = None
        dropout_47 = torch.nn.functional.dropout(x_184, 0.1, False, False)
        x_184 = None
        residual_26 = residual_25 + dropout_47
        residual_25 = dropout_47 = None
        x_185 = torch.nn.functional.layer_norm(
            residual_26,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_186 = torch._C._nn.linear(
            x_185,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_185 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_187 = torch.nn.functional.silu(x_186, inplace=False)
        x_186 = None
        x_188 = torch.nn.functional.dropout(x_187, 0.1, False, False)
        x_187 = None
        x_189 = torch._C._nn.linear(
            x_188,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_188 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_49 = torch.nn.functional.dropout(x_189, 0.1, False, False)
        x_189 = None
        mul_13 = dropout_49 * 0.5
        dropout_49 = None
        residual_27 = residual_26 + mul_13
        residual_26 = mul_13 = None
        x_190 = torch.nn.functional.layer_norm(
            residual_27,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_27 = (
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_ = None
        x_191 = torch.nn.functional.layer_norm(
            x_190,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_192 = torch._C._nn.linear(
            x_191,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_191 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_193 = torch.nn.functional.silu(x_192, inplace=False)
        x_192 = None
        x_194 = torch.nn.functional.dropout(x_193, 0.1, False, False)
        x_193 = None
        x_195 = torch._C._nn.linear(
            x_194,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_194 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_51 = torch.nn.functional.dropout(x_195, 0.1, False, False)
        x_195 = None
        mul_14 = dropout_51 * 0.5
        dropout_51 = None
        residual_28 = x_190 + mul_14
        x_190 = mul_14 = None
        x_196 = torch.nn.functional.layer_norm(
            residual_28,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_
        ) = None
        linear_66 = torch._C._nn.linear(
            x_196,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_21 = linear_66.view(1, -1, 8, 64)
        linear_66 = None
        linear_67 = torch._C._nn.linear(
            x_196,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_14 = linear_67.view(1, -1, 8, 64)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            x_196,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_196 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_14 = linear_68.view(1, -1, 8, 64)
        linear_68 = None
        q_22 = q_21.transpose(1, 2)
        q_21 = None
        k_15 = k_14.transpose(1, 2)
        k_14 = None
        v_15 = v_14.transpose(1, 2)
        v_14 = None
        q_23 = q_22.transpose(1, 2)
        q_22 = None
        linear_69 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_14 = linear_69.view(1, -1, 8, 64)
        linear_69 = None
        p_15 = p_14.transpose(1, 2)
        p_14 = None
        add_56 = (
            q_23
            + l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_7 = add_56.transpose(1, 2)
        add_56 = None
        add_57 = (
            q_23
            + l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_
        )
        q_23 = (
            l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_7 = add_57.transpose(1, 2)
        add_57 = None
        transpose_94 = p_15.transpose(-2, -1)
        p_15 = None
        matrix_bd_14 = torch.matmul(q_with_bias_v_7, transpose_94)
        q_with_bias_v_7 = transpose_94 = None
        x_197 = torch._C._nn.pad(matrix_bd_14, (1, 0), "constant", None)
        matrix_bd_14 = None
        x_198 = x_197.view(1, 8, -1, 66)
        x_197 = None
        getitem_16 = x_198[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_198 = None
        x_199 = getitem_16.view(1, 8, 66, 131)
        getitem_16 = None
        transpose_95 = k_15.transpose(-2, -1)
        k_15 = None
        matrix_ac_7 = torch.matmul(q_with_bias_u_7, transpose_95)
        q_with_bias_u_7 = transpose_95 = None
        matrix_bd_15 = x_199[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_199 = None
        add_58 = matrix_ac_7 + matrix_bd_15
        matrix_ac_7 = matrix_bd_15 = None
        scores_14 = add_58 / 8.0
        add_58 = None
        mask_7 = att_mask_3.unsqueeze(1)
        scores_15 = scores_14.masked_fill(mask_7, -10000.0)
        scores_14 = None
        softmax_7 = torch.softmax(scores_15, dim=-1)
        scores_15 = None
        attn_7 = softmax_7.masked_fill(mask_7, 0.0)
        softmax_7 = mask_7 = None
        p_attn_7 = torch.nn.functional.dropout(attn_7, 0.1, False, False)
        attn_7 = None
        x_200 = torch.matmul(p_attn_7, v_15)
        p_attn_7 = v_15 = None
        transpose_96 = x_200.transpose(1, 2)
        x_200 = None
        x_201 = transpose_96.reshape(1, -1, 512)
        transpose_96 = None
        out_7 = torch._C._nn.linear(
            x_201,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_201 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_53 = torch.nn.functional.dropout(out_7, 0.1, False, False)
        out_7 = None
        residual_29 = residual_28 + dropout_53
        residual_28 = dropout_53 = None
        x_202 = torch.nn.functional.layer_norm(
            residual_29,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_
        ) = None
        x_203 = x_202.transpose(1, 2)
        x_202 = None
        x_204 = torch.conv1d(
            x_203,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_203 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_205 = torch.nn.functional.glu(x_204, dim=1)
        x_204 = None
        unsqueeze_18 = pad_mask_1.unsqueeze(1)
        x_206 = x_205.masked_fill(unsqueeze_18, 0.0)
        x_205 = unsqueeze_18 = None
        new_x_7 = torch._C._nn.pad(x_206, (4, 4), "constant", None)
        x_206 = None
        x_207 = torch.conv1d(
            new_x_7,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_7 = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_209 = torch.nn.functional.silu(x_208, inplace=False)
        x_208 = None
        x_210 = torch.conv1d(
            x_209,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_209 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_211 = x_210.transpose(1, 2)
        x_210 = None
        dropout_54 = torch.nn.functional.dropout(x_211, 0.1, False, False)
        x_211 = None
        residual_30 = residual_29 + dropout_54
        residual_29 = dropout_54 = None
        x_212 = torch.nn.functional.layer_norm(
            residual_30,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_213 = torch._C._nn.linear(
            x_212,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_212 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_214 = torch.nn.functional.silu(x_213, inplace=False)
        x_213 = None
        x_215 = torch.nn.functional.dropout(x_214, 0.1, False, False)
        x_214 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_215 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_56 = torch.nn.functional.dropout(x_216, 0.1, False, False)
        x_216 = None
        mul_15 = dropout_56 * 0.5
        dropout_56 = None
        residual_31 = residual_30 + mul_15
        residual_30 = mul_15 = None
        x_217 = torch.nn.functional.layer_norm(
            residual_31,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_31 = (
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_ = None
        x_218 = torch.nn.functional.layer_norm(
            x_217,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_219 = torch._C._nn.linear(
            x_218,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_218 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_220 = torch.nn.functional.silu(x_219, inplace=False)
        x_219 = None
        x_221 = torch.nn.functional.dropout(x_220, 0.1, False, False)
        x_220 = None
        x_222 = torch._C._nn.linear(
            x_221,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_221 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_58 = torch.nn.functional.dropout(x_222, 0.1, False, False)
        x_222 = None
        mul_16 = dropout_58 * 0.5
        dropout_58 = None
        residual_32 = x_217 + mul_16
        x_217 = mul_16 = None
        x_223 = torch.nn.functional.layer_norm(
            residual_32,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_
        ) = None
        linear_75 = torch._C._nn.linear(
            x_223,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_24 = linear_75.view(1, -1, 8, 64)
        linear_75 = None
        linear_76 = torch._C._nn.linear(
            x_223,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_16 = linear_76.view(1, -1, 8, 64)
        linear_76 = None
        linear_77 = torch._C._nn.linear(
            x_223,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_223 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_16 = linear_77.view(1, -1, 8, 64)
        linear_77 = None
        q_25 = q_24.transpose(1, 2)
        q_24 = None
        k_17 = k_16.transpose(1, 2)
        k_16 = None
        v_17 = v_16.transpose(1, 2)
        v_16 = None
        q_26 = q_25.transpose(1, 2)
        q_25 = None
        linear_78 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_16 = linear_78.view(1, -1, 8, 64)
        linear_78 = None
        p_17 = p_16.transpose(1, 2)
        p_16 = None
        add_63 = (
            q_26
            + l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_8 = add_63.transpose(1, 2)
        add_63 = None
        add_64 = (
            q_26
            + l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_
        )
        q_26 = (
            l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_8 = add_64.transpose(1, 2)
        add_64 = None
        transpose_106 = p_17.transpose(-2, -1)
        p_17 = None
        matrix_bd_16 = torch.matmul(q_with_bias_v_8, transpose_106)
        q_with_bias_v_8 = transpose_106 = None
        x_224 = torch._C._nn.pad(matrix_bd_16, (1, 0), "constant", None)
        matrix_bd_16 = None
        x_225 = x_224.view(1, 8, -1, 66)
        x_224 = None
        getitem_18 = x_225[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_225 = None
        x_226 = getitem_18.view(1, 8, 66, 131)
        getitem_18 = None
        transpose_107 = k_17.transpose(-2, -1)
        k_17 = None
        matrix_ac_8 = torch.matmul(q_with_bias_u_8, transpose_107)
        q_with_bias_u_8 = transpose_107 = None
        matrix_bd_17 = x_226[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_226 = None
        add_65 = matrix_ac_8 + matrix_bd_17
        matrix_ac_8 = matrix_bd_17 = None
        scores_16 = add_65 / 8.0
        add_65 = None
        mask_8 = att_mask_3.unsqueeze(1)
        scores_17 = scores_16.masked_fill(mask_8, -10000.0)
        scores_16 = None
        softmax_8 = torch.softmax(scores_17, dim=-1)
        scores_17 = None
        attn_8 = softmax_8.masked_fill(mask_8, 0.0)
        softmax_8 = mask_8 = None
        p_attn_8 = torch.nn.functional.dropout(attn_8, 0.1, False, False)
        attn_8 = None
        x_227 = torch.matmul(p_attn_8, v_17)
        p_attn_8 = v_17 = None
        transpose_108 = x_227.transpose(1, 2)
        x_227 = None
        x_228 = transpose_108.reshape(1, -1, 512)
        transpose_108 = None
        out_8 = torch._C._nn.linear(
            x_228,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_228 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_60 = torch.nn.functional.dropout(out_8, 0.1, False, False)
        out_8 = None
        residual_33 = residual_32 + dropout_60
        residual_32 = dropout_60 = None
        x_229 = torch.nn.functional.layer_norm(
            residual_33,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_
        ) = None
        x_230 = x_229.transpose(1, 2)
        x_229 = None
        x_231 = torch.conv1d(
            x_230,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_230 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_232 = torch.nn.functional.glu(x_231, dim=1)
        x_231 = None
        unsqueeze_20 = pad_mask_1.unsqueeze(1)
        x_233 = x_232.masked_fill(unsqueeze_20, 0.0)
        x_232 = unsqueeze_20 = None
        new_x_8 = torch._C._nn.pad(x_233, (4, 4), "constant", None)
        x_233 = None
        x_234 = torch.conv1d(
            new_x_8,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_8 = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_236 = torch.nn.functional.silu(x_235, inplace=False)
        x_235 = None
        x_237 = torch.conv1d(
            x_236,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_236 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_238 = x_237.transpose(1, 2)
        x_237 = None
        dropout_61 = torch.nn.functional.dropout(x_238, 0.1, False, False)
        x_238 = None
        residual_34 = residual_33 + dropout_61
        residual_33 = dropout_61 = None
        x_239 = torch.nn.functional.layer_norm(
            residual_34,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_240 = torch._C._nn.linear(
            x_239,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_239 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_241 = torch.nn.functional.silu(x_240, inplace=False)
        x_240 = None
        x_242 = torch.nn.functional.dropout(x_241, 0.1, False, False)
        x_241 = None
        x_243 = torch._C._nn.linear(
            x_242,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_242 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_63 = torch.nn.functional.dropout(x_243, 0.1, False, False)
        x_243 = None
        mul_17 = dropout_63 * 0.5
        dropout_63 = None
        residual_35 = residual_34 + mul_17
        residual_34 = mul_17 = None
        x_244 = torch.nn.functional.layer_norm(
            residual_35,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_35 = (
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_ = None
        x_245 = torch.nn.functional.layer_norm(
            x_244,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_246 = torch._C._nn.linear(
            x_245,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_245 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_247 = torch.nn.functional.silu(x_246, inplace=False)
        x_246 = None
        x_248 = torch.nn.functional.dropout(x_247, 0.1, False, False)
        x_247 = None
        x_249 = torch._C._nn.linear(
            x_248,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_248 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_65 = torch.nn.functional.dropout(x_249, 0.1, False, False)
        x_249 = None
        mul_18 = dropout_65 * 0.5
        dropout_65 = None
        residual_36 = x_244 + mul_18
        x_244 = mul_18 = None
        x_250 = torch.nn.functional.layer_norm(
            residual_36,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            x_250,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_27 = linear_84.view(1, -1, 8, 64)
        linear_84 = None
        linear_85 = torch._C._nn.linear(
            x_250,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_18 = linear_85.view(1, -1, 8, 64)
        linear_85 = None
        linear_86 = torch._C._nn.linear(
            x_250,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_250 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_18 = linear_86.view(1, -1, 8, 64)
        linear_86 = None
        q_28 = q_27.transpose(1, 2)
        q_27 = None
        k_19 = k_18.transpose(1, 2)
        k_18 = None
        v_19 = v_18.transpose(1, 2)
        v_18 = None
        q_29 = q_28.transpose(1, 2)
        q_28 = None
        linear_87 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_18 = linear_87.view(1, -1, 8, 64)
        linear_87 = None
        p_19 = p_18.transpose(1, 2)
        p_18 = None
        add_70 = (
            q_29
            + l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_9 = add_70.transpose(1, 2)
        add_70 = None
        add_71 = (
            q_29
            + l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_
        )
        q_29 = (
            l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_9 = add_71.transpose(1, 2)
        add_71 = None
        transpose_118 = p_19.transpose(-2, -1)
        p_19 = None
        matrix_bd_18 = torch.matmul(q_with_bias_v_9, transpose_118)
        q_with_bias_v_9 = transpose_118 = None
        x_251 = torch._C._nn.pad(matrix_bd_18, (1, 0), "constant", None)
        matrix_bd_18 = None
        x_252 = x_251.view(1, 8, -1, 66)
        x_251 = None
        getitem_20 = x_252[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_252 = None
        x_253 = getitem_20.view(1, 8, 66, 131)
        getitem_20 = None
        transpose_119 = k_19.transpose(-2, -1)
        k_19 = None
        matrix_ac_9 = torch.matmul(q_with_bias_u_9, transpose_119)
        q_with_bias_u_9 = transpose_119 = None
        matrix_bd_19 = x_253[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_253 = None
        add_72 = matrix_ac_9 + matrix_bd_19
        matrix_ac_9 = matrix_bd_19 = None
        scores_18 = add_72 / 8.0
        add_72 = None
        mask_9 = att_mask_3.unsqueeze(1)
        scores_19 = scores_18.masked_fill(mask_9, -10000.0)
        scores_18 = None
        softmax_9 = torch.softmax(scores_19, dim=-1)
        scores_19 = None
        attn_9 = softmax_9.masked_fill(mask_9, 0.0)
        softmax_9 = mask_9 = None
        p_attn_9 = torch.nn.functional.dropout(attn_9, 0.1, False, False)
        attn_9 = None
        x_254 = torch.matmul(p_attn_9, v_19)
        p_attn_9 = v_19 = None
        transpose_120 = x_254.transpose(1, 2)
        x_254 = None
        x_255 = transpose_120.reshape(1, -1, 512)
        transpose_120 = None
        out_9 = torch._C._nn.linear(
            x_255,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_255 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_67 = torch.nn.functional.dropout(out_9, 0.1, False, False)
        out_9 = None
        residual_37 = residual_36 + dropout_67
        residual_36 = dropout_67 = None
        x_256 = torch.nn.functional.layer_norm(
            residual_37,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_
        ) = None
        x_257 = x_256.transpose(1, 2)
        x_256 = None
        x_258 = torch.conv1d(
            x_257,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_257 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_259 = torch.nn.functional.glu(x_258, dim=1)
        x_258 = None
        unsqueeze_22 = pad_mask_1.unsqueeze(1)
        x_260 = x_259.masked_fill(unsqueeze_22, 0.0)
        x_259 = unsqueeze_22 = None
        new_x_9 = torch._C._nn.pad(x_260, (4, 4), "constant", None)
        x_260 = None
        x_261 = torch.conv1d(
            new_x_9,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_9 = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_262 = torch.nn.functional.batch_norm(
            x_261,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_261 = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_263 = torch.nn.functional.silu(x_262, inplace=False)
        x_262 = None
        x_264 = torch.conv1d(
            x_263,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_263 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_265 = x_264.transpose(1, 2)
        x_264 = None
        dropout_68 = torch.nn.functional.dropout(x_265, 0.1, False, False)
        x_265 = None
        residual_38 = residual_37 + dropout_68
        residual_37 = dropout_68 = None
        x_266 = torch.nn.functional.layer_norm(
            residual_38,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_267 = torch._C._nn.linear(
            x_266,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_266 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_268 = torch.nn.functional.silu(x_267, inplace=False)
        x_267 = None
        x_269 = torch.nn.functional.dropout(x_268, 0.1, False, False)
        x_268 = None
        x_270 = torch._C._nn.linear(
            x_269,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_269 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_70 = torch.nn.functional.dropout(x_270, 0.1, False, False)
        x_270 = None
        mul_19 = dropout_70 * 0.5
        dropout_70 = None
        residual_39 = residual_38 + mul_19
        residual_38 = mul_19 = None
        x_271 = torch.nn.functional.layer_norm(
            residual_39,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_39 = (
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_ = None
        x_272 = torch.nn.functional.layer_norm(
            x_271,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_273 = torch._C._nn.linear(
            x_272,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_272 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_274 = torch.nn.functional.silu(x_273, inplace=False)
        x_273 = None
        x_275 = torch.nn.functional.dropout(x_274, 0.1, False, False)
        x_274 = None
        x_276 = torch._C._nn.linear(
            x_275,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_275 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_72 = torch.nn.functional.dropout(x_276, 0.1, False, False)
        x_276 = None
        mul_20 = dropout_72 * 0.5
        dropout_72 = None
        residual_40 = x_271 + mul_20
        x_271 = mul_20 = None
        x_277 = torch.nn.functional.layer_norm(
            residual_40,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_
        ) = None
        linear_93 = torch._C._nn.linear(
            x_277,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_30 = linear_93.view(1, -1, 8, 64)
        linear_93 = None
        linear_94 = torch._C._nn.linear(
            x_277,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_20 = linear_94.view(1, -1, 8, 64)
        linear_94 = None
        linear_95 = torch._C._nn.linear(
            x_277,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_277 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_20 = linear_95.view(1, -1, 8, 64)
        linear_95 = None
        q_31 = q_30.transpose(1, 2)
        q_30 = None
        k_21 = k_20.transpose(1, 2)
        k_20 = None
        v_21 = v_20.transpose(1, 2)
        v_20 = None
        q_32 = q_31.transpose(1, 2)
        q_31 = None
        linear_96 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_20 = linear_96.view(1, -1, 8, 64)
        linear_96 = None
        p_21 = p_20.transpose(1, 2)
        p_20 = None
        add_77 = (
            q_32
            + l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_10 = add_77.transpose(1, 2)
        add_77 = None
        add_78 = (
            q_32
            + l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_
        )
        q_32 = l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_10 = add_78.transpose(1, 2)
        add_78 = None
        transpose_130 = p_21.transpose(-2, -1)
        p_21 = None
        matrix_bd_20 = torch.matmul(q_with_bias_v_10, transpose_130)
        q_with_bias_v_10 = transpose_130 = None
        x_278 = torch._C._nn.pad(matrix_bd_20, (1, 0), "constant", None)
        matrix_bd_20 = None
        x_279 = x_278.view(1, 8, -1, 66)
        x_278 = None
        getitem_22 = x_279[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_279 = None
        x_280 = getitem_22.view(1, 8, 66, 131)
        getitem_22 = None
        transpose_131 = k_21.transpose(-2, -1)
        k_21 = None
        matrix_ac_10 = torch.matmul(q_with_bias_u_10, transpose_131)
        q_with_bias_u_10 = transpose_131 = None
        matrix_bd_21 = x_280[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_280 = None
        add_79 = matrix_ac_10 + matrix_bd_21
        matrix_ac_10 = matrix_bd_21 = None
        scores_20 = add_79 / 8.0
        add_79 = None
        mask_10 = att_mask_3.unsqueeze(1)
        scores_21 = scores_20.masked_fill(mask_10, -10000.0)
        scores_20 = None
        softmax_10 = torch.softmax(scores_21, dim=-1)
        scores_21 = None
        attn_10 = softmax_10.masked_fill(mask_10, 0.0)
        softmax_10 = mask_10 = None
        p_attn_10 = torch.nn.functional.dropout(attn_10, 0.1, False, False)
        attn_10 = None
        x_281 = torch.matmul(p_attn_10, v_21)
        p_attn_10 = v_21 = None
        transpose_132 = x_281.transpose(1, 2)
        x_281 = None
        x_282 = transpose_132.reshape(1, -1, 512)
        transpose_132 = None
        out_10 = torch._C._nn.linear(
            x_282,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_282 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_74 = torch.nn.functional.dropout(out_10, 0.1, False, False)
        out_10 = None
        residual_41 = residual_40 + dropout_74
        residual_40 = dropout_74 = None
        x_283 = torch.nn.functional.layer_norm(
            residual_41,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_
        ) = None
        x_284 = x_283.transpose(1, 2)
        x_283 = None
        x_285 = torch.conv1d(
            x_284,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_284 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_286 = torch.nn.functional.glu(x_285, dim=1)
        x_285 = None
        unsqueeze_24 = pad_mask_1.unsqueeze(1)
        x_287 = x_286.masked_fill(unsqueeze_24, 0.0)
        x_286 = unsqueeze_24 = None
        new_x_10 = torch._C._nn.pad(x_287, (4, 4), "constant", None)
        x_287 = None
        x_288 = torch.conv1d(
            new_x_10,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_10 = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_288 = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_290 = torch.nn.functional.silu(x_289, inplace=False)
        x_289 = None
        x_291 = torch.conv1d(
            x_290,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_290 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_292 = x_291.transpose(1, 2)
        x_291 = None
        dropout_75 = torch.nn.functional.dropout(x_292, 0.1, False, False)
        x_292 = None
        residual_42 = residual_41 + dropout_75
        residual_41 = dropout_75 = None
        x_293 = torch.nn.functional.layer_norm(
            residual_42,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_294 = torch._C._nn.linear(
            x_293,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_293 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_295 = torch.nn.functional.silu(x_294, inplace=False)
        x_294 = None
        x_296 = torch.nn.functional.dropout(x_295, 0.1, False, False)
        x_295 = None
        x_297 = torch._C._nn.linear(
            x_296,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_296 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_77 = torch.nn.functional.dropout(x_297, 0.1, False, False)
        x_297 = None
        mul_21 = dropout_77 * 0.5
        dropout_77 = None
        residual_43 = residual_42 + mul_21
        residual_42 = mul_21 = None
        x_298 = torch.nn.functional.layer_norm(
            residual_43,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_43 = (
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_
        ) = None
        x_299 = torch.nn.functional.layer_norm(
            x_298,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_300 = torch._C._nn.linear(
            x_299,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_299 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_301 = torch.nn.functional.silu(x_300, inplace=False)
        x_300 = None
        x_302 = torch.nn.functional.dropout(x_301, 0.1, False, False)
        x_301 = None
        x_303 = torch._C._nn.linear(
            x_302,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_302 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_79 = torch.nn.functional.dropout(x_303, 0.1, False, False)
        x_303 = None
        mul_22 = dropout_79 * 0.5
        dropout_79 = None
        residual_44 = x_298 + mul_22
        x_298 = mul_22 = None
        x_304 = torch.nn.functional.layer_norm(
            residual_44,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_
        ) = None
        linear_102 = torch._C._nn.linear(
            x_304,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_33 = linear_102.view(1, -1, 8, 64)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            x_304,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_22 = linear_103.view(1, -1, 8, 64)
        linear_103 = None
        linear_104 = torch._C._nn.linear(
            x_304,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_304 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_22 = linear_104.view(1, -1, 8, 64)
        linear_104 = None
        q_34 = q_33.transpose(1, 2)
        q_33 = None
        k_23 = k_22.transpose(1, 2)
        k_22 = None
        v_23 = v_22.transpose(1, 2)
        v_22 = None
        q_35 = q_34.transpose(1, 2)
        q_34 = None
        linear_105 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_22 = linear_105.view(1, -1, 8, 64)
        linear_105 = None
        p_23 = p_22.transpose(1, 2)
        p_22 = None
        add_84 = (
            q_35
            + l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_11 = add_84.transpose(1, 2)
        add_84 = None
        add_85 = (
            q_35
            + l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_
        )
        q_35 = l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_11 = add_85.transpose(1, 2)
        add_85 = None
        transpose_142 = p_23.transpose(-2, -1)
        p_23 = None
        matrix_bd_22 = torch.matmul(q_with_bias_v_11, transpose_142)
        q_with_bias_v_11 = transpose_142 = None
        x_305 = torch._C._nn.pad(matrix_bd_22, (1, 0), "constant", None)
        matrix_bd_22 = None
        x_306 = x_305.view(1, 8, -1, 66)
        x_305 = None
        getitem_24 = x_306[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_306 = None
        x_307 = getitem_24.view(1, 8, 66, 131)
        getitem_24 = None
        transpose_143 = k_23.transpose(-2, -1)
        k_23 = None
        matrix_ac_11 = torch.matmul(q_with_bias_u_11, transpose_143)
        q_with_bias_u_11 = transpose_143 = None
        matrix_bd_23 = x_307[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_307 = None
        add_86 = matrix_ac_11 + matrix_bd_23
        matrix_ac_11 = matrix_bd_23 = None
        scores_22 = add_86 / 8.0
        add_86 = None
        mask_11 = att_mask_3.unsqueeze(1)
        scores_23 = scores_22.masked_fill(mask_11, -10000.0)
        scores_22 = None
        softmax_11 = torch.softmax(scores_23, dim=-1)
        scores_23 = None
        attn_11 = softmax_11.masked_fill(mask_11, 0.0)
        softmax_11 = mask_11 = None
        p_attn_11 = torch.nn.functional.dropout(attn_11, 0.1, False, False)
        attn_11 = None
        x_308 = torch.matmul(p_attn_11, v_23)
        p_attn_11 = v_23 = None
        transpose_144 = x_308.transpose(1, 2)
        x_308 = None
        x_309 = transpose_144.reshape(1, -1, 512)
        transpose_144 = None
        out_11 = torch._C._nn.linear(
            x_309,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_309 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_81 = torch.nn.functional.dropout(out_11, 0.1, False, False)
        out_11 = None
        residual_45 = residual_44 + dropout_81
        residual_44 = dropout_81 = None
        x_310 = torch.nn.functional.layer_norm(
            residual_45,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_
        ) = None
        x_311 = x_310.transpose(1, 2)
        x_310 = None
        x_312 = torch.conv1d(
            x_311,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_311 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_313 = torch.nn.functional.glu(x_312, dim=1)
        x_312 = None
        unsqueeze_26 = pad_mask_1.unsqueeze(1)
        x_314 = x_313.masked_fill(unsqueeze_26, 0.0)
        x_313 = unsqueeze_26 = None
        new_x_11 = torch._C._nn.pad(x_314, (4, 4), "constant", None)
        x_314 = None
        x_315 = torch.conv1d(
            new_x_11,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_11 = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_315 = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_317 = torch.nn.functional.silu(x_316, inplace=False)
        x_316 = None
        x_318 = torch.conv1d(
            x_317,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_317 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_319 = x_318.transpose(1, 2)
        x_318 = None
        dropout_82 = torch.nn.functional.dropout(x_319, 0.1, False, False)
        x_319 = None
        residual_46 = residual_45 + dropout_82
        residual_45 = dropout_82 = None
        x_320 = torch.nn.functional.layer_norm(
            residual_46,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_321 = torch._C._nn.linear(
            x_320,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_320 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_322 = torch.nn.functional.silu(x_321, inplace=False)
        x_321 = None
        x_323 = torch.nn.functional.dropout(x_322, 0.1, False, False)
        x_322 = None
        x_324 = torch._C._nn.linear(
            x_323,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_323 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_84 = torch.nn.functional.dropout(x_324, 0.1, False, False)
        x_324 = None
        mul_23 = dropout_84 * 0.5
        dropout_84 = None
        residual_47 = residual_46 + mul_23
        residual_46 = mul_23 = None
        x_325 = torch.nn.functional.layer_norm(
            residual_47,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_47 = (
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_
        ) = None
        x_326 = torch.nn.functional.layer_norm(
            x_325,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_327 = torch._C._nn.linear(
            x_326,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_326 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_328 = torch.nn.functional.silu(x_327, inplace=False)
        x_327 = None
        x_329 = torch.nn.functional.dropout(x_328, 0.1, False, False)
        x_328 = None
        x_330 = torch._C._nn.linear(
            x_329,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_329 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_86 = torch.nn.functional.dropout(x_330, 0.1, False, False)
        x_330 = None
        mul_24 = dropout_86 * 0.5
        dropout_86 = None
        residual_48 = x_325 + mul_24
        x_325 = mul_24 = None
        x_331 = torch.nn.functional.layer_norm(
            residual_48,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_
        ) = None
        linear_111 = torch._C._nn.linear(
            x_331,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_36 = linear_111.view(1, -1, 8, 64)
        linear_111 = None
        linear_112 = torch._C._nn.linear(
            x_331,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_24 = linear_112.view(1, -1, 8, 64)
        linear_112 = None
        linear_113 = torch._C._nn.linear(
            x_331,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_331 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_24 = linear_113.view(1, -1, 8, 64)
        linear_113 = None
        q_37 = q_36.transpose(1, 2)
        q_36 = None
        k_25 = k_24.transpose(1, 2)
        k_24 = None
        v_25 = v_24.transpose(1, 2)
        v_24 = None
        q_38 = q_37.transpose(1, 2)
        q_37 = None
        linear_114 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_24 = linear_114.view(1, -1, 8, 64)
        linear_114 = None
        p_25 = p_24.transpose(1, 2)
        p_24 = None
        add_91 = (
            q_38
            + l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_12 = add_91.transpose(1, 2)
        add_91 = None
        add_92 = (
            q_38
            + l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_
        )
        q_38 = l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_12 = add_92.transpose(1, 2)
        add_92 = None
        transpose_154 = p_25.transpose(-2, -1)
        p_25 = None
        matrix_bd_24 = torch.matmul(q_with_bias_v_12, transpose_154)
        q_with_bias_v_12 = transpose_154 = None
        x_332 = torch._C._nn.pad(matrix_bd_24, (1, 0), "constant", None)
        matrix_bd_24 = None
        x_333 = x_332.view(1, 8, -1, 66)
        x_332 = None
        getitem_26 = x_333[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_333 = None
        x_334 = getitem_26.view(1, 8, 66, 131)
        getitem_26 = None
        transpose_155 = k_25.transpose(-2, -1)
        k_25 = None
        matrix_ac_12 = torch.matmul(q_with_bias_u_12, transpose_155)
        q_with_bias_u_12 = transpose_155 = None
        matrix_bd_25 = x_334[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_334 = None
        add_93 = matrix_ac_12 + matrix_bd_25
        matrix_ac_12 = matrix_bd_25 = None
        scores_24 = add_93 / 8.0
        add_93 = None
        mask_12 = att_mask_3.unsqueeze(1)
        scores_25 = scores_24.masked_fill(mask_12, -10000.0)
        scores_24 = None
        softmax_12 = torch.softmax(scores_25, dim=-1)
        scores_25 = None
        attn_12 = softmax_12.masked_fill(mask_12, 0.0)
        softmax_12 = mask_12 = None
        p_attn_12 = torch.nn.functional.dropout(attn_12, 0.1, False, False)
        attn_12 = None
        x_335 = torch.matmul(p_attn_12, v_25)
        p_attn_12 = v_25 = None
        transpose_156 = x_335.transpose(1, 2)
        x_335 = None
        x_336 = transpose_156.reshape(1, -1, 512)
        transpose_156 = None
        out_12 = torch._C._nn.linear(
            x_336,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_336 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_88 = torch.nn.functional.dropout(out_12, 0.1, False, False)
        out_12 = None
        residual_49 = residual_48 + dropout_88
        residual_48 = dropout_88 = None
        x_337 = torch.nn.functional.layer_norm(
            residual_49,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_
        ) = None
        x_338 = x_337.transpose(1, 2)
        x_337 = None
        x_339 = torch.conv1d(
            x_338,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_338 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_340 = torch.nn.functional.glu(x_339, dim=1)
        x_339 = None
        unsqueeze_28 = pad_mask_1.unsqueeze(1)
        x_341 = x_340.masked_fill(unsqueeze_28, 0.0)
        x_340 = unsqueeze_28 = None
        new_x_12 = torch._C._nn.pad(x_341, (4, 4), "constant", None)
        x_341 = None
        x_342 = torch.conv1d(
            new_x_12,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_12 = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_343 = torch.nn.functional.batch_norm(
            x_342,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_342 = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_344 = torch.nn.functional.silu(x_343, inplace=False)
        x_343 = None
        x_345 = torch.conv1d(
            x_344,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_344 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_346 = x_345.transpose(1, 2)
        x_345 = None
        dropout_89 = torch.nn.functional.dropout(x_346, 0.1, False, False)
        x_346 = None
        residual_50 = residual_49 + dropout_89
        residual_49 = dropout_89 = None
        x_347 = torch.nn.functional.layer_norm(
            residual_50,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_348 = torch._C._nn.linear(
            x_347,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_347 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_349 = torch.nn.functional.silu(x_348, inplace=False)
        x_348 = None
        x_350 = torch.nn.functional.dropout(x_349, 0.1, False, False)
        x_349 = None
        x_351 = torch._C._nn.linear(
            x_350,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_350 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_91 = torch.nn.functional.dropout(x_351, 0.1, False, False)
        x_351 = None
        mul_25 = dropout_91 * 0.5
        dropout_91 = None
        residual_51 = residual_50 + mul_25
        residual_50 = mul_25 = None
        x_352 = torch.nn.functional.layer_norm(
            residual_51,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_51 = (
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_
        ) = None
        x_353 = torch.nn.functional.layer_norm(
            x_352,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_354 = torch._C._nn.linear(
            x_353,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_353 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_355 = torch.nn.functional.silu(x_354, inplace=False)
        x_354 = None
        x_356 = torch.nn.functional.dropout(x_355, 0.1, False, False)
        x_355 = None
        x_357 = torch._C._nn.linear(
            x_356,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_356 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_93 = torch.nn.functional.dropout(x_357, 0.1, False, False)
        x_357 = None
        mul_26 = dropout_93 * 0.5
        dropout_93 = None
        residual_52 = x_352 + mul_26
        x_352 = mul_26 = None
        x_358 = torch.nn.functional.layer_norm(
            residual_52,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_
        ) = None
        linear_120 = torch._C._nn.linear(
            x_358,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_39 = linear_120.view(1, -1, 8, 64)
        linear_120 = None
        linear_121 = torch._C._nn.linear(
            x_358,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_26 = linear_121.view(1, -1, 8, 64)
        linear_121 = None
        linear_122 = torch._C._nn.linear(
            x_358,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_358 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_26 = linear_122.view(1, -1, 8, 64)
        linear_122 = None
        q_40 = q_39.transpose(1, 2)
        q_39 = None
        k_27 = k_26.transpose(1, 2)
        k_26 = None
        v_27 = v_26.transpose(1, 2)
        v_26 = None
        q_41 = q_40.transpose(1, 2)
        q_40 = None
        linear_123 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_26 = linear_123.view(1, -1, 8, 64)
        linear_123 = None
        p_27 = p_26.transpose(1, 2)
        p_26 = None
        add_98 = (
            q_41
            + l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_13 = add_98.transpose(1, 2)
        add_98 = None
        add_99 = (
            q_41
            + l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_
        )
        q_41 = l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_13 = add_99.transpose(1, 2)
        add_99 = None
        transpose_166 = p_27.transpose(-2, -1)
        p_27 = None
        matrix_bd_26 = torch.matmul(q_with_bias_v_13, transpose_166)
        q_with_bias_v_13 = transpose_166 = None
        x_359 = torch._C._nn.pad(matrix_bd_26, (1, 0), "constant", None)
        matrix_bd_26 = None
        x_360 = x_359.view(1, 8, -1, 66)
        x_359 = None
        getitem_28 = x_360[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_360 = None
        x_361 = getitem_28.view(1, 8, 66, 131)
        getitem_28 = None
        transpose_167 = k_27.transpose(-2, -1)
        k_27 = None
        matrix_ac_13 = torch.matmul(q_with_bias_u_13, transpose_167)
        q_with_bias_u_13 = transpose_167 = None
        matrix_bd_27 = x_361[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_361 = None
        add_100 = matrix_ac_13 + matrix_bd_27
        matrix_ac_13 = matrix_bd_27 = None
        scores_26 = add_100 / 8.0
        add_100 = None
        mask_13 = att_mask_3.unsqueeze(1)
        scores_27 = scores_26.masked_fill(mask_13, -10000.0)
        scores_26 = None
        softmax_13 = torch.softmax(scores_27, dim=-1)
        scores_27 = None
        attn_13 = softmax_13.masked_fill(mask_13, 0.0)
        softmax_13 = mask_13 = None
        p_attn_13 = torch.nn.functional.dropout(attn_13, 0.1, False, False)
        attn_13 = None
        x_362 = torch.matmul(p_attn_13, v_27)
        p_attn_13 = v_27 = None
        transpose_168 = x_362.transpose(1, 2)
        x_362 = None
        x_363 = transpose_168.reshape(1, -1, 512)
        transpose_168 = None
        out_13 = torch._C._nn.linear(
            x_363,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_363 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_95 = torch.nn.functional.dropout(out_13, 0.1, False, False)
        out_13 = None
        residual_53 = residual_52 + dropout_95
        residual_52 = dropout_95 = None
        x_364 = torch.nn.functional.layer_norm(
            residual_53,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_
        ) = None
        x_365 = x_364.transpose(1, 2)
        x_364 = None
        x_366 = torch.conv1d(
            x_365,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_365 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_367 = torch.nn.functional.glu(x_366, dim=1)
        x_366 = None
        unsqueeze_30 = pad_mask_1.unsqueeze(1)
        x_368 = x_367.masked_fill(unsqueeze_30, 0.0)
        x_367 = unsqueeze_30 = None
        new_x_13 = torch._C._nn.pad(x_368, (4, 4), "constant", None)
        x_368 = None
        x_369 = torch.conv1d(
            new_x_13,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_13 = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_370 = torch.nn.functional.batch_norm(
            x_369,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_369 = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_371 = torch.nn.functional.silu(x_370, inplace=False)
        x_370 = None
        x_372 = torch.conv1d(
            x_371,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_371 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_373 = x_372.transpose(1, 2)
        x_372 = None
        dropout_96 = torch.nn.functional.dropout(x_373, 0.1, False, False)
        x_373 = None
        residual_54 = residual_53 + dropout_96
        residual_53 = dropout_96 = None
        x_374 = torch.nn.functional.layer_norm(
            residual_54,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_375 = torch._C._nn.linear(
            x_374,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_374 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_376 = torch.nn.functional.silu(x_375, inplace=False)
        x_375 = None
        x_377 = torch.nn.functional.dropout(x_376, 0.1, False, False)
        x_376 = None
        x_378 = torch._C._nn.linear(
            x_377,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_377 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_98 = torch.nn.functional.dropout(x_378, 0.1, False, False)
        x_378 = None
        mul_27 = dropout_98 * 0.5
        dropout_98 = None
        residual_55 = residual_54 + mul_27
        residual_54 = mul_27 = None
        x_379 = torch.nn.functional.layer_norm(
            residual_55,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_55 = (
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_
        ) = None
        x_380 = torch.nn.functional.layer_norm(
            x_379,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_381 = torch._C._nn.linear(
            x_380,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_380 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_382 = torch.nn.functional.silu(x_381, inplace=False)
        x_381 = None
        x_383 = torch.nn.functional.dropout(x_382, 0.1, False, False)
        x_382 = None
        x_384 = torch._C._nn.linear(
            x_383,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_383 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_100 = torch.nn.functional.dropout(x_384, 0.1, False, False)
        x_384 = None
        mul_28 = dropout_100 * 0.5
        dropout_100 = None
        residual_56 = x_379 + mul_28
        x_379 = mul_28 = None
        x_385 = torch.nn.functional.layer_norm(
            residual_56,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_
        ) = None
        linear_129 = torch._C._nn.linear(
            x_385,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_42 = linear_129.view(1, -1, 8, 64)
        linear_129 = None
        linear_130 = torch._C._nn.linear(
            x_385,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_28 = linear_130.view(1, -1, 8, 64)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            x_385,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_385 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_28 = linear_131.view(1, -1, 8, 64)
        linear_131 = None
        q_43 = q_42.transpose(1, 2)
        q_42 = None
        k_29 = k_28.transpose(1, 2)
        k_28 = None
        v_29 = v_28.transpose(1, 2)
        v_28 = None
        q_44 = q_43.transpose(1, 2)
        q_43 = None
        linear_132 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_28 = linear_132.view(1, -1, 8, 64)
        linear_132 = None
        p_29 = p_28.transpose(1, 2)
        p_28 = None
        add_105 = (
            q_44
            + l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_14 = add_105.transpose(1, 2)
        add_105 = None
        add_106 = (
            q_44
            + l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_
        )
        q_44 = l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_14 = add_106.transpose(1, 2)
        add_106 = None
        transpose_178 = p_29.transpose(-2, -1)
        p_29 = None
        matrix_bd_28 = torch.matmul(q_with_bias_v_14, transpose_178)
        q_with_bias_v_14 = transpose_178 = None
        x_386 = torch._C._nn.pad(matrix_bd_28, (1, 0), "constant", None)
        matrix_bd_28 = None
        x_387 = x_386.view(1, 8, -1, 66)
        x_386 = None
        getitem_30 = x_387[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_387 = None
        x_388 = getitem_30.view(1, 8, 66, 131)
        getitem_30 = None
        transpose_179 = k_29.transpose(-2, -1)
        k_29 = None
        matrix_ac_14 = torch.matmul(q_with_bias_u_14, transpose_179)
        q_with_bias_u_14 = transpose_179 = None
        matrix_bd_29 = x_388[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_388 = None
        add_107 = matrix_ac_14 + matrix_bd_29
        matrix_ac_14 = matrix_bd_29 = None
        scores_28 = add_107 / 8.0
        add_107 = None
        mask_14 = att_mask_3.unsqueeze(1)
        scores_29 = scores_28.masked_fill(mask_14, -10000.0)
        scores_28 = None
        softmax_14 = torch.softmax(scores_29, dim=-1)
        scores_29 = None
        attn_14 = softmax_14.masked_fill(mask_14, 0.0)
        softmax_14 = mask_14 = None
        p_attn_14 = torch.nn.functional.dropout(attn_14, 0.1, False, False)
        attn_14 = None
        x_389 = torch.matmul(p_attn_14, v_29)
        p_attn_14 = v_29 = None
        transpose_180 = x_389.transpose(1, 2)
        x_389 = None
        x_390 = transpose_180.reshape(1, -1, 512)
        transpose_180 = None
        out_14 = torch._C._nn.linear(
            x_390,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_390 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_102 = torch.nn.functional.dropout(out_14, 0.1, False, False)
        out_14 = None
        residual_57 = residual_56 + dropout_102
        residual_56 = dropout_102 = None
        x_391 = torch.nn.functional.layer_norm(
            residual_57,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_
        ) = None
        x_392 = x_391.transpose(1, 2)
        x_391 = None
        x_393 = torch.conv1d(
            x_392,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_392 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_394 = torch.nn.functional.glu(x_393, dim=1)
        x_393 = None
        unsqueeze_32 = pad_mask_1.unsqueeze(1)
        x_395 = x_394.masked_fill(unsqueeze_32, 0.0)
        x_394 = unsqueeze_32 = None
        new_x_14 = torch._C._nn.pad(x_395, (4, 4), "constant", None)
        x_395 = None
        x_396 = torch.conv1d(
            new_x_14,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_14 = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_397 = torch.nn.functional.batch_norm(
            x_396,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_396 = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_398 = torch.nn.functional.silu(x_397, inplace=False)
        x_397 = None
        x_399 = torch.conv1d(
            x_398,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_398 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_400 = x_399.transpose(1, 2)
        x_399 = None
        dropout_103 = torch.nn.functional.dropout(x_400, 0.1, False, False)
        x_400 = None
        residual_58 = residual_57 + dropout_103
        residual_57 = dropout_103 = None
        x_401 = torch.nn.functional.layer_norm(
            residual_58,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_402 = torch._C._nn.linear(
            x_401,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_401 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_403 = torch.nn.functional.silu(x_402, inplace=False)
        x_402 = None
        x_404 = torch.nn.functional.dropout(x_403, 0.1, False, False)
        x_403 = None
        x_405 = torch._C._nn.linear(
            x_404,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_404 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_105 = torch.nn.functional.dropout(x_405, 0.1, False, False)
        x_405 = None
        mul_29 = dropout_105 * 0.5
        dropout_105 = None
        residual_59 = residual_58 + mul_29
        residual_58 = mul_29 = None
        x_406 = torch.nn.functional.layer_norm(
            residual_59,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_59 = (
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_
        ) = None
        x_407 = torch.nn.functional.layer_norm(
            x_406,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_408 = torch._C._nn.linear(
            x_407,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_407 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_409 = torch.nn.functional.silu(x_408, inplace=False)
        x_408 = None
        x_410 = torch.nn.functional.dropout(x_409, 0.1, False, False)
        x_409 = None
        x_411 = torch._C._nn.linear(
            x_410,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_410 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_107 = torch.nn.functional.dropout(x_411, 0.1, False, False)
        x_411 = None
        mul_30 = dropout_107 * 0.5
        dropout_107 = None
        residual_60 = x_406 + mul_30
        x_406 = mul_30 = None
        x_412 = torch.nn.functional.layer_norm(
            residual_60,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_
        ) = None
        linear_138 = torch._C._nn.linear(
            x_412,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_45 = linear_138.view(1, -1, 8, 64)
        linear_138 = None
        linear_139 = torch._C._nn.linear(
            x_412,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_30 = linear_139.view(1, -1, 8, 64)
        linear_139 = None
        linear_140 = torch._C._nn.linear(
            x_412,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_412 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_30 = linear_140.view(1, -1, 8, 64)
        linear_140 = None
        q_46 = q_45.transpose(1, 2)
        q_45 = None
        k_31 = k_30.transpose(1, 2)
        k_30 = None
        v_31 = v_30.transpose(1, 2)
        v_30 = None
        q_47 = q_46.transpose(1, 2)
        q_46 = None
        linear_141 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_ = (
            None
        )
        p_30 = linear_141.view(1, -1, 8, 64)
        linear_141 = None
        p_31 = p_30.transpose(1, 2)
        p_30 = None
        add_112 = (
            q_47
            + l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_15 = add_112.transpose(1, 2)
        add_112 = None
        add_113 = (
            q_47
            + l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_
        )
        q_47 = l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_15 = add_113.transpose(1, 2)
        add_113 = None
        transpose_190 = p_31.transpose(-2, -1)
        p_31 = None
        matrix_bd_30 = torch.matmul(q_with_bias_v_15, transpose_190)
        q_with_bias_v_15 = transpose_190 = None
        x_413 = torch._C._nn.pad(matrix_bd_30, (1, 0), "constant", None)
        matrix_bd_30 = None
        x_414 = x_413.view(1, 8, -1, 66)
        x_413 = None
        getitem_32 = x_414[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_414 = None
        x_415 = getitem_32.view(1, 8, 66, 131)
        getitem_32 = None
        transpose_191 = k_31.transpose(-2, -1)
        k_31 = None
        matrix_ac_15 = torch.matmul(q_with_bias_u_15, transpose_191)
        q_with_bias_u_15 = transpose_191 = None
        matrix_bd_31 = x_415[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_415 = None
        add_114 = matrix_ac_15 + matrix_bd_31
        matrix_ac_15 = matrix_bd_31 = None
        scores_30 = add_114 / 8.0
        add_114 = None
        mask_15 = att_mask_3.unsqueeze(1)
        scores_31 = scores_30.masked_fill(mask_15, -10000.0)
        scores_30 = None
        softmax_15 = torch.softmax(scores_31, dim=-1)
        scores_31 = None
        attn_15 = softmax_15.masked_fill(mask_15, 0.0)
        softmax_15 = mask_15 = None
        p_attn_15 = torch.nn.functional.dropout(attn_15, 0.1, False, False)
        attn_15 = None
        x_416 = torch.matmul(p_attn_15, v_31)
        p_attn_15 = v_31 = None
        transpose_192 = x_416.transpose(1, 2)
        x_416 = None
        x_417 = transpose_192.reshape(1, -1, 512)
        transpose_192 = None
        out_15 = torch._C._nn.linear(
            x_417,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_417 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_109 = torch.nn.functional.dropout(out_15, 0.1, False, False)
        out_15 = None
        residual_61 = residual_60 + dropout_109
        residual_60 = dropout_109 = None
        x_418 = torch.nn.functional.layer_norm(
            residual_61,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_
        ) = None
        x_419 = x_418.transpose(1, 2)
        x_418 = None
        x_420 = torch.conv1d(
            x_419,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_419 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_421 = torch.nn.functional.glu(x_420, dim=1)
        x_420 = None
        unsqueeze_34 = pad_mask_1.unsqueeze(1)
        x_422 = x_421.masked_fill(unsqueeze_34, 0.0)
        x_421 = unsqueeze_34 = None
        new_x_15 = torch._C._nn.pad(x_422, (4, 4), "constant", None)
        x_422 = None
        x_423 = torch.conv1d(
            new_x_15,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_15 = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_424 = torch.nn.functional.batch_norm(
            x_423,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_423 = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_425 = torch.nn.functional.silu(x_424, inplace=False)
        x_424 = None
        x_426 = torch.conv1d(
            x_425,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_425 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_427 = x_426.transpose(1, 2)
        x_426 = None
        dropout_110 = torch.nn.functional.dropout(x_427, 0.1, False, False)
        x_427 = None
        residual_62 = residual_61 + dropout_110
        residual_61 = dropout_110 = None
        x_428 = torch.nn.functional.layer_norm(
            residual_62,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_429 = torch._C._nn.linear(
            x_428,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_428 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_430 = torch.nn.functional.silu(x_429, inplace=False)
        x_429 = None
        x_431 = torch.nn.functional.dropout(x_430, 0.1, False, False)
        x_430 = None
        x_432 = torch._C._nn.linear(
            x_431,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_431 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_112 = torch.nn.functional.dropout(x_432, 0.1, False, False)
        x_432 = None
        mul_31 = dropout_112 * 0.5
        dropout_112 = None
        residual_63 = residual_62 + mul_31
        residual_62 = mul_31 = None
        x_433 = torch.nn.functional.layer_norm(
            residual_63,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_63 = (
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_
        ) = None
        x_434 = torch.nn.functional.layer_norm(
            x_433,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_435 = torch._C._nn.linear(
            x_434,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_434 = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_436 = torch.nn.functional.silu(x_435, inplace=False)
        x_435 = None
        x_437 = torch.nn.functional.dropout(x_436, 0.1, False, False)
        x_436 = None
        x_438 = torch._C._nn.linear(
            x_437,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_437 = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_114 = torch.nn.functional.dropout(x_438, 0.1, False, False)
        x_438 = None
        mul_32 = dropout_114 * 0.5
        dropout_114 = None
        residual_64 = x_433 + mul_32
        x_433 = mul_32 = None
        x_439 = torch.nn.functional.layer_norm(
            residual_64,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_16_modules_norm_self_att_parameters_bias_
        ) = None
        linear_147 = torch._C._nn.linear(
            x_439,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_48 = linear_147.view(1, -1, 8, 64)
        linear_147 = None
        linear_148 = torch._C._nn.linear(
            x_439,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_32 = linear_148.view(1, -1, 8, 64)
        linear_148 = None
        linear_149 = torch._C._nn.linear(
            x_439,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_439 = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_32 = linear_149.view(1, -1, 8, 64)
        linear_149 = None
        q_49 = q_48.transpose(1, 2)
        q_48 = None
        k_33 = k_32.transpose(1, 2)
        k_32 = None
        v_33 = v_32.transpose(1, 2)
        v_32 = None
        q_50 = q_49.transpose(1, 2)
        q_49 = None
        linear_150 = torch._C._nn.linear(
            pos_emb,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_,
            None,
        )
        pos_emb = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_pos_parameters_weight_ = (None)
        p_32 = linear_150.view(1, -1, 8, 64)
        linear_150 = None
        p_33 = p_32.transpose(1, 2)
        p_32 = None
        add_119 = (
            q_50
            + l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_16 = add_119.transpose(1, 2)
        add_119 = None
        add_120 = (
            q_50
            + l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_
        )
        q_50 = l_instance_modules_layers_modules_16_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_16 = add_120.transpose(1, 2)
        add_120 = None
        transpose_202 = p_33.transpose(-2, -1)
        p_33 = None
        matrix_bd_32 = torch.matmul(q_with_bias_v_16, transpose_202)
        q_with_bias_v_16 = transpose_202 = None
        x_440 = torch._C._nn.pad(matrix_bd_32, (1, 0), "constant", None)
        matrix_bd_32 = None
        x_441 = x_440.view(1, 8, -1, 66)
        x_440 = None
        getitem_34 = x_441[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_441 = None
        x_442 = getitem_34.view(1, 8, 66, 131)
        getitem_34 = None
        transpose_203 = k_33.transpose(-2, -1)
        k_33 = None
        matrix_ac_16 = torch.matmul(q_with_bias_u_16, transpose_203)
        q_with_bias_u_16 = transpose_203 = None
        matrix_bd_33 = x_442[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_442 = None
        add_121 = matrix_ac_16 + matrix_bd_33
        matrix_ac_16 = matrix_bd_33 = None
        scores_32 = add_121 / 8.0
        add_121 = None
        mask_16 = att_mask_3.unsqueeze(1)
        att_mask_3 = None
        scores_33 = scores_32.masked_fill(mask_16, -10000.0)
        scores_32 = None
        softmax_16 = torch.softmax(scores_33, dim=-1)
        scores_33 = None
        attn_16 = softmax_16.masked_fill(mask_16, 0.0)
        softmax_16 = mask_16 = None
        p_attn_16 = torch.nn.functional.dropout(attn_16, 0.1, False, False)
        attn_16 = None
        x_443 = torch.matmul(p_attn_16, v_33)
        p_attn_16 = v_33 = None
        transpose_204 = x_443.transpose(1, 2)
        x_443 = None
        x_444 = transpose_204.reshape(1, -1, 512)
        transpose_204 = None
        out_16 = torch._C._nn.linear(
            x_444,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_444 = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_116 = torch.nn.functional.dropout(out_16, 0.1, False, False)
        out_16 = None
        residual_65 = residual_64 + dropout_116
        residual_64 = dropout_116 = None
        x_445 = torch.nn.functional.layer_norm(
            residual_65,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_
        ) = None
        x_446 = x_445.transpose(1, 2)
        x_445 = None
        x_447 = torch.conv1d(
            x_446,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_446 = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_448 = torch.nn.functional.glu(x_447, dim=1)
        x_447 = None
        unsqueeze_36 = pad_mask_1.unsqueeze(1)
        pad_mask_1 = None
        x_449 = x_448.masked_fill(unsqueeze_36, 0.0)
        x_448 = unsqueeze_36 = None
        new_x_16 = torch._C._nn.pad(x_449, (4, 4), "constant", None)
        x_449 = None
        x_450 = torch.conv1d(
            new_x_16,
            l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_16 = l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_451 = torch.nn.functional.batch_norm(
            x_450,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_450 = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_452 = torch.nn.functional.silu(x_451, inplace=False)
        x_451 = None
        x_453 = torch.conv1d(
            x_452,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_452 = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_454 = x_453.transpose(1, 2)
        x_453 = None
        dropout_117 = torch.nn.functional.dropout(x_454, 0.1, False, False)
        x_454 = None
        residual_66 = residual_65 + dropout_117
        residual_65 = dropout_117 = None
        x_455 = torch.nn.functional.layer_norm(
            residual_66,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_456 = torch._C._nn.linear(
            x_455,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_455 = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_457 = torch.nn.functional.silu(x_456, inplace=False)
        x_456 = None
        x_458 = torch.nn.functional.dropout(x_457, 0.1, False, False)
        x_457 = None
        x_459 = torch._C._nn.linear(
            x_458,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_458 = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_119 = torch.nn.functional.dropout(x_459, 0.1, False, False)
        x_459 = None
        mul_33 = dropout_119 * 0.5
        dropout_119 = None
        residual_67 = residual_66 + mul_33
        residual_66 = mul_33 = None
        x_460 = torch.nn.functional.layer_norm(
            residual_67,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_67 = (
            l_instance_modules_layers_modules_16_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_16_modules_norm_out_parameters_bias_
        ) = None
        audio_signal_2 = torch.transpose(x_460, 1, 2)
        x_460 = None
        length_1 = length.to(dtype=torch.int64)
        length = None
        return (audio_signal_2, length_1)
