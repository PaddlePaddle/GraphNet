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
        input_2 = torch.nn.functional.relu(input_1, inplace=True)
        input_1 = None
        input_3 = torch.conv2d(
            input_2,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        input_2 = (
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_ = None
        input_4 = torch.nn.functional.relu(input_3, inplace=True)
        input_3 = None
        transpose_1 = input_4.transpose(1, 2)
        input_4 = None
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
        length = lengths_4.to(torch.int64)
        lengths_4 = None
        x_2 = x_1 * 13.2664991614216
        x_1 = None
        pos_emb = l_instance_modules_pos_enc_buffers_pe_[
            (slice(None, None, None), slice(4869, 5130, None))
        ]
        l_instance_modules_pos_enc_buffers_pe_ = None
        audio_signal_1 = torch.nn.functional.dropout(x_2, 0.1, False, False)
        x_2 = None
        att_mask = torch.ones(
            1, 131, 131, dtype=torch.bool, device=device(type="cuda", index=0)
        )
        arange = torch.arange(0, 131, device=device(type="cuda", index=0))
        expand = arange.expand(1, -1)
        arange = None
        unsqueeze_1 = length.unsqueeze(-1)
        pad_mask = expand < unsqueeze_1
        expand = unsqueeze_1 = None
        unsqueeze_2 = pad_mask.unsqueeze(1)
        pad_mask_for_att_mask = unsqueeze_2.repeat([1, 131, 1])
        unsqueeze_2 = None
        transpose_2 = pad_mask_for_att_mask.transpose(1, 2)
        pad_mask_for_att_mask_1 = torch.logical_and(pad_mask_for_att_mask, transpose_2)
        pad_mask_for_att_mask = transpose_2 = None
        att_mask_1 = att_mask[
            (slice(None, None, None), slice(None, 131, None), slice(None, 131, None))
        ]
        att_mask = None
        to_4 = att_mask_1.to(device(type="cuda", index=0))
        att_mask_1 = None
        att_mask_2 = torch.logical_and(pad_mask_for_att_mask_1, to_4)
        pad_mask_for_att_mask_1 = to_4 = None
        att_mask_3 = ~att_mask_2
        att_mask_2 = None
        pad_mask_1 = ~pad_mask
        pad_mask = None
        x_3 = torch.nn.functional.layer_norm(
            audio_signal_1,
            (176,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_4 = torch._C._nn.linear(
            x_3,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_3 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_5 = torch.nn.functional.silu(x_4, inplace=False)
        x_4 = None
        x_6 = torch.nn.functional.dropout(x_5, 0.1, False, False)
        x_5 = None
        x_7 = torch._C._nn.linear(
            x_6,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_6 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_2 = torch.nn.functional.dropout(x_7, 0.1, False, False)
        x_7 = None
        mul_1 = dropout_2 * 0.5
        dropout_2 = None
        residual = audio_signal_1 + mul_1
        audio_signal_1 = mul_1 = None
        x_8 = torch.nn.functional.layer_norm(
            residual,
            (176,),
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_0_modules_norm_self_att_parameters_bias_
        ) = None
        linear_3 = torch._C._nn.linear(
            x_8,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q = linear_3.view(1, -1, 4, 44)
        linear_3 = None
        linear_4 = torch._C._nn.linear(
            x_8,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k = linear_4.view(1, -1, 4, 44)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            x_8,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_8 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v = linear_5.view(1, -1, 4, 44)
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
        p = linear_6.view(1, -1, 4, 44)
        linear_6 = None
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
        x_9 = torch._C._nn.pad(matrix_bd, (1, 0), "constant", None)
        matrix_bd = None
        x_10 = x_9.view(1, 4, -1, 131)
        x_9 = None
        getitem_2 = x_10[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_10 = None
        x_11 = getitem_2.view(1, 4, 131, 261)
        getitem_2 = None
        transpose_11 = k_1.transpose(-2, -1)
        k_1 = None
        matrix_ac = torch.matmul(q_with_bias_u, transpose_11)
        q_with_bias_u = transpose_11 = None
        matrix_bd_1 = x_11[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_11 = None
        add_7 = matrix_ac + matrix_bd_1
        matrix_ac = matrix_bd_1 = None
        scores = add_7 / 6.6332495807108
        add_7 = None
        mask = att_mask_3.unsqueeze(1)
        scores_1 = scores.masked_fill(mask, -10000.0)
        scores = None
        softmax = torch.softmax(scores_1, dim=-1)
        scores_1 = None
        attn = softmax.masked_fill(mask, 0.0)
        softmax = mask = None
        p_attn = torch.nn.functional.dropout(attn, 0.1, False, False)
        attn = None
        x_12 = torch.matmul(p_attn, v_1)
        p_attn = v_1 = None
        transpose_12 = x_12.transpose(1, 2)
        x_12 = None
        x_13 = transpose_12.reshape(1, -1, 176)
        transpose_12 = None
        out = torch._C._nn.linear(
            x_13,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_13 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_4 = torch.nn.functional.dropout(out, 0.1, False, False)
        out = None
        residual_1 = residual + dropout_4
        residual = dropout_4 = None
        x_14 = torch.nn.functional.layer_norm(
            residual_1,
            (176,),
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_
        ) = None
        x_15 = x_14.transpose(1, 2)
        x_14 = None
        x_16 = torch.conv1d(
            x_15,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_15 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_17 = torch.nn.functional.glu(x_16, dim=1)
        x_16 = None
        unsqueeze_4 = pad_mask_1.unsqueeze(1)
        x_18 = x_17.masked_fill(unsqueeze_4, 0.0)
        x_17 = unsqueeze_4 = None
        new_x = torch._C._nn.pad(x_18, (15, 15), "constant", None)
        x_18 = None
        x_19 = torch.conv1d(
            new_x,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_21 = torch.nn.functional.silu(x_20, inplace=False)
        x_20 = None
        x_22 = torch.conv1d(
            x_21,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_21 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_23 = x_22.transpose(1, 2)
        x_22 = None
        dropout_5 = torch.nn.functional.dropout(x_23, 0.1, False, False)
        x_23 = None
        residual_2 = residual_1 + dropout_5
        residual_1 = dropout_5 = None
        x_24 = torch.nn.functional.layer_norm(
            residual_2,
            (176,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_25 = torch._C._nn.linear(
            x_24,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_24 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_26 = torch.nn.functional.silu(x_25, inplace=False)
        x_25 = None
        x_27 = torch.nn.functional.dropout(x_26, 0.1, False, False)
        x_26 = None
        x_28 = torch._C._nn.linear(
            x_27,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_27 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_7 = torch.nn.functional.dropout(x_28, 0.1, False, False)
        x_28 = None
        mul_2 = dropout_7 * 0.5
        dropout_7 = None
        residual_3 = residual_2 + mul_2
        residual_2 = mul_2 = None
        x_29 = torch.nn.functional.layer_norm(
            residual_3,
            (176,),
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_3 = (
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_ = None
        x_30 = torch.nn.functional.layer_norm(
            x_29,
            (176,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_31 = torch._C._nn.linear(
            x_30,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_30 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_32 = torch.nn.functional.silu(x_31, inplace=False)
        x_31 = None
        x_33 = torch.nn.functional.dropout(x_32, 0.1, False, False)
        x_32 = None
        x_34 = torch._C._nn.linear(
            x_33,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_33 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_9 = torch.nn.functional.dropout(x_34, 0.1, False, False)
        x_34 = None
        mul_3 = dropout_9 * 0.5
        dropout_9 = None
        residual_4 = x_29 + mul_3
        x_29 = mul_3 = None
        x_35 = torch.nn.functional.layer_norm(
            residual_4,
            (176,),
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_1_modules_norm_self_att_parameters_bias_
        ) = None
        linear_12 = torch._C._nn.linear(
            x_35,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_3 = linear_12.view(1, -1, 4, 44)
        linear_12 = None
        linear_13 = torch._C._nn.linear(
            x_35,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_2 = linear_13.view(1, -1, 4, 44)
        linear_13 = None
        linear_14 = torch._C._nn.linear(
            x_35,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_35 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_2 = linear_14.view(1, -1, 4, 44)
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
        p_2 = linear_15.view(1, -1, 4, 44)
        linear_15 = None
        p_3 = p_2.transpose(1, 2)
        p_2 = None
        add_12 = (
            q_5
            + l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_1 = add_12.transpose(1, 2)
        add_12 = None
        add_13 = (
            q_5
            + l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_
        )
        q_5 = (
            l_instance_modules_layers_modules_1_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_1 = add_13.transpose(1, 2)
        add_13 = None
        transpose_22 = p_3.transpose(-2, -1)
        p_3 = None
        matrix_bd_2 = torch.matmul(q_with_bias_v_1, transpose_22)
        q_with_bias_v_1 = transpose_22 = None
        x_36 = torch._C._nn.pad(matrix_bd_2, (1, 0), "constant", None)
        matrix_bd_2 = None
        x_37 = x_36.view(1, 4, -1, 131)
        x_36 = None
        getitem_4 = x_37[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_37 = None
        x_38 = getitem_4.view(1, 4, 131, 261)
        getitem_4 = None
        transpose_23 = k_3.transpose(-2, -1)
        k_3 = None
        matrix_ac_1 = torch.matmul(q_with_bias_u_1, transpose_23)
        q_with_bias_u_1 = transpose_23 = None
        matrix_bd_3 = x_38[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_38 = None
        add_14 = matrix_ac_1 + matrix_bd_3
        matrix_ac_1 = matrix_bd_3 = None
        scores_2 = add_14 / 6.6332495807108
        add_14 = None
        mask_1 = att_mask_3.unsqueeze(1)
        scores_3 = scores_2.masked_fill(mask_1, -10000.0)
        scores_2 = None
        softmax_1 = torch.softmax(scores_3, dim=-1)
        scores_3 = None
        attn_1 = softmax_1.masked_fill(mask_1, 0.0)
        softmax_1 = mask_1 = None
        p_attn_1 = torch.nn.functional.dropout(attn_1, 0.1, False, False)
        attn_1 = None
        x_39 = torch.matmul(p_attn_1, v_3)
        p_attn_1 = v_3 = None
        transpose_24 = x_39.transpose(1, 2)
        x_39 = None
        x_40 = transpose_24.reshape(1, -1, 176)
        transpose_24 = None
        out_1 = torch._C._nn.linear(
            x_40,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_40 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_11 = torch.nn.functional.dropout(out_1, 0.1, False, False)
        out_1 = None
        residual_5 = residual_4 + dropout_11
        residual_4 = dropout_11 = None
        x_41 = torch.nn.functional.layer_norm(
            residual_5,
            (176,),
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_
        ) = None
        x_42 = x_41.transpose(1, 2)
        x_41 = None
        x_43 = torch.conv1d(
            x_42,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_42 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_44 = torch.nn.functional.glu(x_43, dim=1)
        x_43 = None
        unsqueeze_6 = pad_mask_1.unsqueeze(1)
        x_45 = x_44.masked_fill(unsqueeze_6, 0.0)
        x_44 = unsqueeze_6 = None
        new_x_1 = torch._C._nn.pad(x_45, (15, 15), "constant", None)
        x_45 = None
        x_46 = torch.conv1d(
            new_x_1,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_1 = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_48 = torch.nn.functional.silu(x_47, inplace=False)
        x_47 = None
        x_49 = torch.conv1d(
            x_48,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_48 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_50 = x_49.transpose(1, 2)
        x_49 = None
        dropout_12 = torch.nn.functional.dropout(x_50, 0.1, False, False)
        x_50 = None
        residual_6 = residual_5 + dropout_12
        residual_5 = dropout_12 = None
        x_51 = torch.nn.functional.layer_norm(
            residual_6,
            (176,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_52 = torch._C._nn.linear(
            x_51,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_51 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_53 = torch.nn.functional.silu(x_52, inplace=False)
        x_52 = None
        x_54 = torch.nn.functional.dropout(x_53, 0.1, False, False)
        x_53 = None
        x_55 = torch._C._nn.linear(
            x_54,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_54 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_14 = torch.nn.functional.dropout(x_55, 0.1, False, False)
        x_55 = None
        mul_4 = dropout_14 * 0.5
        dropout_14 = None
        residual_7 = residual_6 + mul_4
        residual_6 = mul_4 = None
        x_56 = torch.nn.functional.layer_norm(
            residual_7,
            (176,),
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_7 = (
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_ = None
        x_57 = torch.nn.functional.layer_norm(
            x_56,
            (176,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_58 = torch._C._nn.linear(
            x_57,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_57 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_59 = torch.nn.functional.silu(x_58, inplace=False)
        x_58 = None
        x_60 = torch.nn.functional.dropout(x_59, 0.1, False, False)
        x_59 = None
        x_61 = torch._C._nn.linear(
            x_60,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_60 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_16 = torch.nn.functional.dropout(x_61, 0.1, False, False)
        x_61 = None
        mul_5 = dropout_16 * 0.5
        dropout_16 = None
        residual_8 = x_56 + mul_5
        x_56 = mul_5 = None
        x_62 = torch.nn.functional.layer_norm(
            residual_8,
            (176,),
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_2_modules_norm_self_att_parameters_bias_
        ) = None
        linear_21 = torch._C._nn.linear(
            x_62,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_6 = linear_21.view(1, -1, 4, 44)
        linear_21 = None
        linear_22 = torch._C._nn.linear(
            x_62,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_4 = linear_22.view(1, -1, 4, 44)
        linear_22 = None
        linear_23 = torch._C._nn.linear(
            x_62,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_62 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_4 = linear_23.view(1, -1, 4, 44)
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
        p_4 = linear_24.view(1, -1, 4, 44)
        linear_24 = None
        p_5 = p_4.transpose(1, 2)
        p_4 = None
        add_19 = (
            q_8
            + l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_2 = add_19.transpose(1, 2)
        add_19 = None
        add_20 = (
            q_8
            + l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_
        )
        q_8 = (
            l_instance_modules_layers_modules_2_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_2 = add_20.transpose(1, 2)
        add_20 = None
        transpose_34 = p_5.transpose(-2, -1)
        p_5 = None
        matrix_bd_4 = torch.matmul(q_with_bias_v_2, transpose_34)
        q_with_bias_v_2 = transpose_34 = None
        x_63 = torch._C._nn.pad(matrix_bd_4, (1, 0), "constant", None)
        matrix_bd_4 = None
        x_64 = x_63.view(1, 4, -1, 131)
        x_63 = None
        getitem_6 = x_64[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_64 = None
        x_65 = getitem_6.view(1, 4, 131, 261)
        getitem_6 = None
        transpose_35 = k_5.transpose(-2, -1)
        k_5 = None
        matrix_ac_2 = torch.matmul(q_with_bias_u_2, transpose_35)
        q_with_bias_u_2 = transpose_35 = None
        matrix_bd_5 = x_65[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_65 = None
        add_21 = matrix_ac_2 + matrix_bd_5
        matrix_ac_2 = matrix_bd_5 = None
        scores_4 = add_21 / 6.6332495807108
        add_21 = None
        mask_2 = att_mask_3.unsqueeze(1)
        scores_5 = scores_4.masked_fill(mask_2, -10000.0)
        scores_4 = None
        softmax_2 = torch.softmax(scores_5, dim=-1)
        scores_5 = None
        attn_2 = softmax_2.masked_fill(mask_2, 0.0)
        softmax_2 = mask_2 = None
        p_attn_2 = torch.nn.functional.dropout(attn_2, 0.1, False, False)
        attn_2 = None
        x_66 = torch.matmul(p_attn_2, v_5)
        p_attn_2 = v_5 = None
        transpose_36 = x_66.transpose(1, 2)
        x_66 = None
        x_67 = transpose_36.reshape(1, -1, 176)
        transpose_36 = None
        out_2 = torch._C._nn.linear(
            x_67,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_67 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_18 = torch.nn.functional.dropout(out_2, 0.1, False, False)
        out_2 = None
        residual_9 = residual_8 + dropout_18
        residual_8 = dropout_18 = None
        x_68 = torch.nn.functional.layer_norm(
            residual_9,
            (176,),
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_
        ) = None
        x_69 = x_68.transpose(1, 2)
        x_68 = None
        x_70 = torch.conv1d(
            x_69,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_69 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_71 = torch.nn.functional.glu(x_70, dim=1)
        x_70 = None
        unsqueeze_8 = pad_mask_1.unsqueeze(1)
        x_72 = x_71.masked_fill(unsqueeze_8, 0.0)
        x_71 = unsqueeze_8 = None
        new_x_2 = torch._C._nn.pad(x_72, (15, 15), "constant", None)
        x_72 = None
        x_73 = torch.conv1d(
            new_x_2,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_2 = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_75 = torch.nn.functional.silu(x_74, inplace=False)
        x_74 = None
        x_76 = torch.conv1d(
            x_75,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_75 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_77 = x_76.transpose(1, 2)
        x_76 = None
        dropout_19 = torch.nn.functional.dropout(x_77, 0.1, False, False)
        x_77 = None
        residual_10 = residual_9 + dropout_19
        residual_9 = dropout_19 = None
        x_78 = torch.nn.functional.layer_norm(
            residual_10,
            (176,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_79 = torch._C._nn.linear(
            x_78,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_78 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_80 = torch.nn.functional.silu(x_79, inplace=False)
        x_79 = None
        x_81 = torch.nn.functional.dropout(x_80, 0.1, False, False)
        x_80 = None
        x_82 = torch._C._nn.linear(
            x_81,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_81 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_21 = torch.nn.functional.dropout(x_82, 0.1, False, False)
        x_82 = None
        mul_6 = dropout_21 * 0.5
        dropout_21 = None
        residual_11 = residual_10 + mul_6
        residual_10 = mul_6 = None
        x_83 = torch.nn.functional.layer_norm(
            residual_11,
            (176,),
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_11 = (
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_ = None
        x_84 = torch.nn.functional.layer_norm(
            x_83,
            (176,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_85 = torch._C._nn.linear(
            x_84,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_84 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_86 = torch.nn.functional.silu(x_85, inplace=False)
        x_85 = None
        x_87 = torch.nn.functional.dropout(x_86, 0.1, False, False)
        x_86 = None
        x_88 = torch._C._nn.linear(
            x_87,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_87 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_23 = torch.nn.functional.dropout(x_88, 0.1, False, False)
        x_88 = None
        mul_7 = dropout_23 * 0.5
        dropout_23 = None
        residual_12 = x_83 + mul_7
        x_83 = mul_7 = None
        x_89 = torch.nn.functional.layer_norm(
            residual_12,
            (176,),
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_3_modules_norm_self_att_parameters_bias_
        ) = None
        linear_30 = torch._C._nn.linear(
            x_89,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_9 = linear_30.view(1, -1, 4, 44)
        linear_30 = None
        linear_31 = torch._C._nn.linear(
            x_89,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_6 = linear_31.view(1, -1, 4, 44)
        linear_31 = None
        linear_32 = torch._C._nn.linear(
            x_89,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_89 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_6 = linear_32.view(1, -1, 4, 44)
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
        p_6 = linear_33.view(1, -1, 4, 44)
        linear_33 = None
        p_7 = p_6.transpose(1, 2)
        p_6 = None
        add_26 = (
            q_11
            + l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_3 = add_26.transpose(1, 2)
        add_26 = None
        add_27 = (
            q_11
            + l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_
        )
        q_11 = (
            l_instance_modules_layers_modules_3_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_3 = add_27.transpose(1, 2)
        add_27 = None
        transpose_46 = p_7.transpose(-2, -1)
        p_7 = None
        matrix_bd_6 = torch.matmul(q_with_bias_v_3, transpose_46)
        q_with_bias_v_3 = transpose_46 = None
        x_90 = torch._C._nn.pad(matrix_bd_6, (1, 0), "constant", None)
        matrix_bd_6 = None
        x_91 = x_90.view(1, 4, -1, 131)
        x_90 = None
        getitem_8 = x_91[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_91 = None
        x_92 = getitem_8.view(1, 4, 131, 261)
        getitem_8 = None
        transpose_47 = k_7.transpose(-2, -1)
        k_7 = None
        matrix_ac_3 = torch.matmul(q_with_bias_u_3, transpose_47)
        q_with_bias_u_3 = transpose_47 = None
        matrix_bd_7 = x_92[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_92 = None
        add_28 = matrix_ac_3 + matrix_bd_7
        matrix_ac_3 = matrix_bd_7 = None
        scores_6 = add_28 / 6.6332495807108
        add_28 = None
        mask_3 = att_mask_3.unsqueeze(1)
        scores_7 = scores_6.masked_fill(mask_3, -10000.0)
        scores_6 = None
        softmax_3 = torch.softmax(scores_7, dim=-1)
        scores_7 = None
        attn_3 = softmax_3.masked_fill(mask_3, 0.0)
        softmax_3 = mask_3 = None
        p_attn_3 = torch.nn.functional.dropout(attn_3, 0.1, False, False)
        attn_3 = None
        x_93 = torch.matmul(p_attn_3, v_7)
        p_attn_3 = v_7 = None
        transpose_48 = x_93.transpose(1, 2)
        x_93 = None
        x_94 = transpose_48.reshape(1, -1, 176)
        transpose_48 = None
        out_3 = torch._C._nn.linear(
            x_94,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_94 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_25 = torch.nn.functional.dropout(out_3, 0.1, False, False)
        out_3 = None
        residual_13 = residual_12 + dropout_25
        residual_12 = dropout_25 = None
        x_95 = torch.nn.functional.layer_norm(
            residual_13,
            (176,),
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_
        ) = None
        x_96 = x_95.transpose(1, 2)
        x_95 = None
        x_97 = torch.conv1d(
            x_96,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_96 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_98 = torch.nn.functional.glu(x_97, dim=1)
        x_97 = None
        unsqueeze_10 = pad_mask_1.unsqueeze(1)
        x_99 = x_98.masked_fill(unsqueeze_10, 0.0)
        x_98 = unsqueeze_10 = None
        new_x_3 = torch._C._nn.pad(x_99, (15, 15), "constant", None)
        x_99 = None
        x_100 = torch.conv1d(
            new_x_3,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_3 = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_101 = torch.nn.functional.batch_norm(
            x_100,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_100 = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_102 = torch.nn.functional.silu(x_101, inplace=False)
        x_101 = None
        x_103 = torch.conv1d(
            x_102,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_102 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_104 = x_103.transpose(1, 2)
        x_103 = None
        dropout_26 = torch.nn.functional.dropout(x_104, 0.1, False, False)
        x_104 = None
        residual_14 = residual_13 + dropout_26
        residual_13 = dropout_26 = None
        x_105 = torch.nn.functional.layer_norm(
            residual_14,
            (176,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_106 = torch._C._nn.linear(
            x_105,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_105 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_107 = torch.nn.functional.silu(x_106, inplace=False)
        x_106 = None
        x_108 = torch.nn.functional.dropout(x_107, 0.1, False, False)
        x_107 = None
        x_109 = torch._C._nn.linear(
            x_108,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_108 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_28 = torch.nn.functional.dropout(x_109, 0.1, False, False)
        x_109 = None
        mul_8 = dropout_28 * 0.5
        dropout_28 = None
        residual_15 = residual_14 + mul_8
        residual_14 = mul_8 = None
        x_110 = torch.nn.functional.layer_norm(
            residual_15,
            (176,),
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_15 = (
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_ = None
        x_111 = torch.nn.functional.layer_norm(
            x_110,
            (176,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_112 = torch._C._nn.linear(
            x_111,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_111 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_113 = torch.nn.functional.silu(x_112, inplace=False)
        x_112 = None
        x_114 = torch.nn.functional.dropout(x_113, 0.1, False, False)
        x_113 = None
        x_115 = torch._C._nn.linear(
            x_114,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_114 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_30 = torch.nn.functional.dropout(x_115, 0.1, False, False)
        x_115 = None
        mul_9 = dropout_30 * 0.5
        dropout_30 = None
        residual_16 = x_110 + mul_9
        x_110 = mul_9 = None
        x_116 = torch.nn.functional.layer_norm(
            residual_16,
            (176,),
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_4_modules_norm_self_att_parameters_bias_
        ) = None
        linear_39 = torch._C._nn.linear(
            x_116,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_12 = linear_39.view(1, -1, 4, 44)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            x_116,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_8 = linear_40.view(1, -1, 4, 44)
        linear_40 = None
        linear_41 = torch._C._nn.linear(
            x_116,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_116 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_8 = linear_41.view(1, -1, 4, 44)
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
        p_8 = linear_42.view(1, -1, 4, 44)
        linear_42 = None
        p_9 = p_8.transpose(1, 2)
        p_8 = None
        add_33 = (
            q_14
            + l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_4 = add_33.transpose(1, 2)
        add_33 = None
        add_34 = (
            q_14
            + l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_
        )
        q_14 = (
            l_instance_modules_layers_modules_4_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_4 = add_34.transpose(1, 2)
        add_34 = None
        transpose_58 = p_9.transpose(-2, -1)
        p_9 = None
        matrix_bd_8 = torch.matmul(q_with_bias_v_4, transpose_58)
        q_with_bias_v_4 = transpose_58 = None
        x_117 = torch._C._nn.pad(matrix_bd_8, (1, 0), "constant", None)
        matrix_bd_8 = None
        x_118 = x_117.view(1, 4, -1, 131)
        x_117 = None
        getitem_10 = x_118[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_118 = None
        x_119 = getitem_10.view(1, 4, 131, 261)
        getitem_10 = None
        transpose_59 = k_9.transpose(-2, -1)
        k_9 = None
        matrix_ac_4 = torch.matmul(q_with_bias_u_4, transpose_59)
        q_with_bias_u_4 = transpose_59 = None
        matrix_bd_9 = x_119[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_119 = None
        add_35 = matrix_ac_4 + matrix_bd_9
        matrix_ac_4 = matrix_bd_9 = None
        scores_8 = add_35 / 6.6332495807108
        add_35 = None
        mask_4 = att_mask_3.unsqueeze(1)
        scores_9 = scores_8.masked_fill(mask_4, -10000.0)
        scores_8 = None
        softmax_4 = torch.softmax(scores_9, dim=-1)
        scores_9 = None
        attn_4 = softmax_4.masked_fill(mask_4, 0.0)
        softmax_4 = mask_4 = None
        p_attn_4 = torch.nn.functional.dropout(attn_4, 0.1, False, False)
        attn_4 = None
        x_120 = torch.matmul(p_attn_4, v_9)
        p_attn_4 = v_9 = None
        transpose_60 = x_120.transpose(1, 2)
        x_120 = None
        x_121 = transpose_60.reshape(1, -1, 176)
        transpose_60 = None
        out_4 = torch._C._nn.linear(
            x_121,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_121 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_32 = torch.nn.functional.dropout(out_4, 0.1, False, False)
        out_4 = None
        residual_17 = residual_16 + dropout_32
        residual_16 = dropout_32 = None
        x_122 = torch.nn.functional.layer_norm(
            residual_17,
            (176,),
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_
        ) = None
        x_123 = x_122.transpose(1, 2)
        x_122 = None
        x_124 = torch.conv1d(
            x_123,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_123 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_125 = torch.nn.functional.glu(x_124, dim=1)
        x_124 = None
        unsqueeze_12 = pad_mask_1.unsqueeze(1)
        x_126 = x_125.masked_fill(unsqueeze_12, 0.0)
        x_125 = unsqueeze_12 = None
        new_x_4 = torch._C._nn.pad(x_126, (15, 15), "constant", None)
        x_126 = None
        x_127 = torch.conv1d(
            new_x_4,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_4 = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_129 = torch.nn.functional.silu(x_128, inplace=False)
        x_128 = None
        x_130 = torch.conv1d(
            x_129,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_129 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_131 = x_130.transpose(1, 2)
        x_130 = None
        dropout_33 = torch.nn.functional.dropout(x_131, 0.1, False, False)
        x_131 = None
        residual_18 = residual_17 + dropout_33
        residual_17 = dropout_33 = None
        x_132 = torch.nn.functional.layer_norm(
            residual_18,
            (176,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_133 = torch._C._nn.linear(
            x_132,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_132 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_134 = torch.nn.functional.silu(x_133, inplace=False)
        x_133 = None
        x_135 = torch.nn.functional.dropout(x_134, 0.1, False, False)
        x_134 = None
        x_136 = torch._C._nn.linear(
            x_135,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_135 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_35 = torch.nn.functional.dropout(x_136, 0.1, False, False)
        x_136 = None
        mul_10 = dropout_35 * 0.5
        dropout_35 = None
        residual_19 = residual_18 + mul_10
        residual_18 = mul_10 = None
        x_137 = torch.nn.functional.layer_norm(
            residual_19,
            (176,),
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_19 = (
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_ = None
        x_138 = torch.nn.functional.layer_norm(
            x_137,
            (176,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_139 = torch._C._nn.linear(
            x_138,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_138 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_140 = torch.nn.functional.silu(x_139, inplace=False)
        x_139 = None
        x_141 = torch.nn.functional.dropout(x_140, 0.1, False, False)
        x_140 = None
        x_142 = torch._C._nn.linear(
            x_141,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_141 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_37 = torch.nn.functional.dropout(x_142, 0.1, False, False)
        x_142 = None
        mul_11 = dropout_37 * 0.5
        dropout_37 = None
        residual_20 = x_137 + mul_11
        x_137 = mul_11 = None
        x_143 = torch.nn.functional.layer_norm(
            residual_20,
            (176,),
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_5_modules_norm_self_att_parameters_bias_
        ) = None
        linear_48 = torch._C._nn.linear(
            x_143,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_15 = linear_48.view(1, -1, 4, 44)
        linear_48 = None
        linear_49 = torch._C._nn.linear(
            x_143,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_10 = linear_49.view(1, -1, 4, 44)
        linear_49 = None
        linear_50 = torch._C._nn.linear(
            x_143,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_143 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_10 = linear_50.view(1, -1, 4, 44)
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
        p_10 = linear_51.view(1, -1, 4, 44)
        linear_51 = None
        p_11 = p_10.transpose(1, 2)
        p_10 = None
        add_40 = (
            q_17
            + l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_5 = add_40.transpose(1, 2)
        add_40 = None
        add_41 = (
            q_17
            + l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_
        )
        q_17 = (
            l_instance_modules_layers_modules_5_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_5 = add_41.transpose(1, 2)
        add_41 = None
        transpose_70 = p_11.transpose(-2, -1)
        p_11 = None
        matrix_bd_10 = torch.matmul(q_with_bias_v_5, transpose_70)
        q_with_bias_v_5 = transpose_70 = None
        x_144 = torch._C._nn.pad(matrix_bd_10, (1, 0), "constant", None)
        matrix_bd_10 = None
        x_145 = x_144.view(1, 4, -1, 131)
        x_144 = None
        getitem_12 = x_145[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_145 = None
        x_146 = getitem_12.view(1, 4, 131, 261)
        getitem_12 = None
        transpose_71 = k_11.transpose(-2, -1)
        k_11 = None
        matrix_ac_5 = torch.matmul(q_with_bias_u_5, transpose_71)
        q_with_bias_u_5 = transpose_71 = None
        matrix_bd_11 = x_146[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_146 = None
        add_42 = matrix_ac_5 + matrix_bd_11
        matrix_ac_5 = matrix_bd_11 = None
        scores_10 = add_42 / 6.6332495807108
        add_42 = None
        mask_5 = att_mask_3.unsqueeze(1)
        scores_11 = scores_10.masked_fill(mask_5, -10000.0)
        scores_10 = None
        softmax_5 = torch.softmax(scores_11, dim=-1)
        scores_11 = None
        attn_5 = softmax_5.masked_fill(mask_5, 0.0)
        softmax_5 = mask_5 = None
        p_attn_5 = torch.nn.functional.dropout(attn_5, 0.1, False, False)
        attn_5 = None
        x_147 = torch.matmul(p_attn_5, v_11)
        p_attn_5 = v_11 = None
        transpose_72 = x_147.transpose(1, 2)
        x_147 = None
        x_148 = transpose_72.reshape(1, -1, 176)
        transpose_72 = None
        out_5 = torch._C._nn.linear(
            x_148,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_148 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_39 = torch.nn.functional.dropout(out_5, 0.1, False, False)
        out_5 = None
        residual_21 = residual_20 + dropout_39
        residual_20 = dropout_39 = None
        x_149 = torch.nn.functional.layer_norm(
            residual_21,
            (176,),
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_
        ) = None
        x_150 = x_149.transpose(1, 2)
        x_149 = None
        x_151 = torch.conv1d(
            x_150,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_150 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_152 = torch.nn.functional.glu(x_151, dim=1)
        x_151 = None
        unsqueeze_14 = pad_mask_1.unsqueeze(1)
        x_153 = x_152.masked_fill(unsqueeze_14, 0.0)
        x_152 = unsqueeze_14 = None
        new_x_5 = torch._C._nn.pad(x_153, (15, 15), "constant", None)
        x_153 = None
        x_154 = torch.conv1d(
            new_x_5,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_5 = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_156 = torch.nn.functional.silu(x_155, inplace=False)
        x_155 = None
        x_157 = torch.conv1d(
            x_156,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_156 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_158 = x_157.transpose(1, 2)
        x_157 = None
        dropout_40 = torch.nn.functional.dropout(x_158, 0.1, False, False)
        x_158 = None
        residual_22 = residual_21 + dropout_40
        residual_21 = dropout_40 = None
        x_159 = torch.nn.functional.layer_norm(
            residual_22,
            (176,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_160 = torch._C._nn.linear(
            x_159,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_159 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_161 = torch.nn.functional.silu(x_160, inplace=False)
        x_160 = None
        x_162 = torch.nn.functional.dropout(x_161, 0.1, False, False)
        x_161 = None
        x_163 = torch._C._nn.linear(
            x_162,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_162 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_42 = torch.nn.functional.dropout(x_163, 0.1, False, False)
        x_163 = None
        mul_12 = dropout_42 * 0.5
        dropout_42 = None
        residual_23 = residual_22 + mul_12
        residual_22 = mul_12 = None
        x_164 = torch.nn.functional.layer_norm(
            residual_23,
            (176,),
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_23 = (
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_ = None
        x_165 = torch.nn.functional.layer_norm(
            x_164,
            (176,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_166 = torch._C._nn.linear(
            x_165,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_165 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_167 = torch.nn.functional.silu(x_166, inplace=False)
        x_166 = None
        x_168 = torch.nn.functional.dropout(x_167, 0.1, False, False)
        x_167 = None
        x_169 = torch._C._nn.linear(
            x_168,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_168 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_44 = torch.nn.functional.dropout(x_169, 0.1, False, False)
        x_169 = None
        mul_13 = dropout_44 * 0.5
        dropout_44 = None
        residual_24 = x_164 + mul_13
        x_164 = mul_13 = None
        x_170 = torch.nn.functional.layer_norm(
            residual_24,
            (176,),
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_6_modules_norm_self_att_parameters_bias_
        ) = None
        linear_57 = torch._C._nn.linear(
            x_170,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_18 = linear_57.view(1, -1, 4, 44)
        linear_57 = None
        linear_58 = torch._C._nn.linear(
            x_170,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_12 = linear_58.view(1, -1, 4, 44)
        linear_58 = None
        linear_59 = torch._C._nn.linear(
            x_170,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_170 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_12 = linear_59.view(1, -1, 4, 44)
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
        p_12 = linear_60.view(1, -1, 4, 44)
        linear_60 = None
        p_13 = p_12.transpose(1, 2)
        p_12 = None
        add_47 = (
            q_20
            + l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_6 = add_47.transpose(1, 2)
        add_47 = None
        add_48 = (
            q_20
            + l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_
        )
        q_20 = (
            l_instance_modules_layers_modules_6_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_6 = add_48.transpose(1, 2)
        add_48 = None
        transpose_82 = p_13.transpose(-2, -1)
        p_13 = None
        matrix_bd_12 = torch.matmul(q_with_bias_v_6, transpose_82)
        q_with_bias_v_6 = transpose_82 = None
        x_171 = torch._C._nn.pad(matrix_bd_12, (1, 0), "constant", None)
        matrix_bd_12 = None
        x_172 = x_171.view(1, 4, -1, 131)
        x_171 = None
        getitem_14 = x_172[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_172 = None
        x_173 = getitem_14.view(1, 4, 131, 261)
        getitem_14 = None
        transpose_83 = k_13.transpose(-2, -1)
        k_13 = None
        matrix_ac_6 = torch.matmul(q_with_bias_u_6, transpose_83)
        q_with_bias_u_6 = transpose_83 = None
        matrix_bd_13 = x_173[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_173 = None
        add_49 = matrix_ac_6 + matrix_bd_13
        matrix_ac_6 = matrix_bd_13 = None
        scores_12 = add_49 / 6.6332495807108
        add_49 = None
        mask_6 = att_mask_3.unsqueeze(1)
        scores_13 = scores_12.masked_fill(mask_6, -10000.0)
        scores_12 = None
        softmax_6 = torch.softmax(scores_13, dim=-1)
        scores_13 = None
        attn_6 = softmax_6.masked_fill(mask_6, 0.0)
        softmax_6 = mask_6 = None
        p_attn_6 = torch.nn.functional.dropout(attn_6, 0.1, False, False)
        attn_6 = None
        x_174 = torch.matmul(p_attn_6, v_13)
        p_attn_6 = v_13 = None
        transpose_84 = x_174.transpose(1, 2)
        x_174 = None
        x_175 = transpose_84.reshape(1, -1, 176)
        transpose_84 = None
        out_6 = torch._C._nn.linear(
            x_175,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_175 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_46 = torch.nn.functional.dropout(out_6, 0.1, False, False)
        out_6 = None
        residual_25 = residual_24 + dropout_46
        residual_24 = dropout_46 = None
        x_176 = torch.nn.functional.layer_norm(
            residual_25,
            (176,),
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_
        ) = None
        x_177 = x_176.transpose(1, 2)
        x_176 = None
        x_178 = torch.conv1d(
            x_177,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_177 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_179 = torch.nn.functional.glu(x_178, dim=1)
        x_178 = None
        unsqueeze_16 = pad_mask_1.unsqueeze(1)
        x_180 = x_179.masked_fill(unsqueeze_16, 0.0)
        x_179 = unsqueeze_16 = None
        new_x_6 = torch._C._nn.pad(x_180, (15, 15), "constant", None)
        x_180 = None
        x_181 = torch.conv1d(
            new_x_6,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_6 = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_181 = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_183 = torch.nn.functional.silu(x_182, inplace=False)
        x_182 = None
        x_184 = torch.conv1d(
            x_183,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_183 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_185 = x_184.transpose(1, 2)
        x_184 = None
        dropout_47 = torch.nn.functional.dropout(x_185, 0.1, False, False)
        x_185 = None
        residual_26 = residual_25 + dropout_47
        residual_25 = dropout_47 = None
        x_186 = torch.nn.functional.layer_norm(
            residual_26,
            (176,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_187 = torch._C._nn.linear(
            x_186,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_186 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_188 = torch.nn.functional.silu(x_187, inplace=False)
        x_187 = None
        x_189 = torch.nn.functional.dropout(x_188, 0.1, False, False)
        x_188 = None
        x_190 = torch._C._nn.linear(
            x_189,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_189 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_49 = torch.nn.functional.dropout(x_190, 0.1, False, False)
        x_190 = None
        mul_14 = dropout_49 * 0.5
        dropout_49 = None
        residual_27 = residual_26 + mul_14
        residual_26 = mul_14 = None
        x_191 = torch.nn.functional.layer_norm(
            residual_27,
            (176,),
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_27 = (
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_ = None
        x_192 = torch.nn.functional.layer_norm(
            x_191,
            (176,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_193 = torch._C._nn.linear(
            x_192,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_192 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_194 = torch.nn.functional.silu(x_193, inplace=False)
        x_193 = None
        x_195 = torch.nn.functional.dropout(x_194, 0.1, False, False)
        x_194 = None
        x_196 = torch._C._nn.linear(
            x_195,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_195 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_51 = torch.nn.functional.dropout(x_196, 0.1, False, False)
        x_196 = None
        mul_15 = dropout_51 * 0.5
        dropout_51 = None
        residual_28 = x_191 + mul_15
        x_191 = mul_15 = None
        x_197 = torch.nn.functional.layer_norm(
            residual_28,
            (176,),
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_7_modules_norm_self_att_parameters_bias_
        ) = None
        linear_66 = torch._C._nn.linear(
            x_197,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_21 = linear_66.view(1, -1, 4, 44)
        linear_66 = None
        linear_67 = torch._C._nn.linear(
            x_197,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_14 = linear_67.view(1, -1, 4, 44)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            x_197,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_197 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_14 = linear_68.view(1, -1, 4, 44)
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
        p_14 = linear_69.view(1, -1, 4, 44)
        linear_69 = None
        p_15 = p_14.transpose(1, 2)
        p_14 = None
        add_54 = (
            q_23
            + l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_7 = add_54.transpose(1, 2)
        add_54 = None
        add_55 = (
            q_23
            + l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_
        )
        q_23 = (
            l_instance_modules_layers_modules_7_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_7 = add_55.transpose(1, 2)
        add_55 = None
        transpose_94 = p_15.transpose(-2, -1)
        p_15 = None
        matrix_bd_14 = torch.matmul(q_with_bias_v_7, transpose_94)
        q_with_bias_v_7 = transpose_94 = None
        x_198 = torch._C._nn.pad(matrix_bd_14, (1, 0), "constant", None)
        matrix_bd_14 = None
        x_199 = x_198.view(1, 4, -1, 131)
        x_198 = None
        getitem_16 = x_199[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_199 = None
        x_200 = getitem_16.view(1, 4, 131, 261)
        getitem_16 = None
        transpose_95 = k_15.transpose(-2, -1)
        k_15 = None
        matrix_ac_7 = torch.matmul(q_with_bias_u_7, transpose_95)
        q_with_bias_u_7 = transpose_95 = None
        matrix_bd_15 = x_200[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_200 = None
        add_56 = matrix_ac_7 + matrix_bd_15
        matrix_ac_7 = matrix_bd_15 = None
        scores_14 = add_56 / 6.6332495807108
        add_56 = None
        mask_7 = att_mask_3.unsqueeze(1)
        scores_15 = scores_14.masked_fill(mask_7, -10000.0)
        scores_14 = None
        softmax_7 = torch.softmax(scores_15, dim=-1)
        scores_15 = None
        attn_7 = softmax_7.masked_fill(mask_7, 0.0)
        softmax_7 = mask_7 = None
        p_attn_7 = torch.nn.functional.dropout(attn_7, 0.1, False, False)
        attn_7 = None
        x_201 = torch.matmul(p_attn_7, v_15)
        p_attn_7 = v_15 = None
        transpose_96 = x_201.transpose(1, 2)
        x_201 = None
        x_202 = transpose_96.reshape(1, -1, 176)
        transpose_96 = None
        out_7 = torch._C._nn.linear(
            x_202,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_202 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_53 = torch.nn.functional.dropout(out_7, 0.1, False, False)
        out_7 = None
        residual_29 = residual_28 + dropout_53
        residual_28 = dropout_53 = None
        x_203 = torch.nn.functional.layer_norm(
            residual_29,
            (176,),
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_
        ) = None
        x_204 = x_203.transpose(1, 2)
        x_203 = None
        x_205 = torch.conv1d(
            x_204,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_204 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_206 = torch.nn.functional.glu(x_205, dim=1)
        x_205 = None
        unsqueeze_18 = pad_mask_1.unsqueeze(1)
        x_207 = x_206.masked_fill(unsqueeze_18, 0.0)
        x_206 = unsqueeze_18 = None
        new_x_7 = torch._C._nn.pad(x_207, (15, 15), "constant", None)
        x_207 = None
        x_208 = torch.conv1d(
            new_x_7,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_7 = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_209 = torch.nn.functional.batch_norm(
            x_208,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_208 = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_210 = torch.nn.functional.silu(x_209, inplace=False)
        x_209 = None
        x_211 = torch.conv1d(
            x_210,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_210 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_212 = x_211.transpose(1, 2)
        x_211 = None
        dropout_54 = torch.nn.functional.dropout(x_212, 0.1, False, False)
        x_212 = None
        residual_30 = residual_29 + dropout_54
        residual_29 = dropout_54 = None
        x_213 = torch.nn.functional.layer_norm(
            residual_30,
            (176,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_214 = torch._C._nn.linear(
            x_213,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_213 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_215 = torch.nn.functional.silu(x_214, inplace=False)
        x_214 = None
        x_216 = torch.nn.functional.dropout(x_215, 0.1, False, False)
        x_215 = None
        x_217 = torch._C._nn.linear(
            x_216,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_216 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_56 = torch.nn.functional.dropout(x_217, 0.1, False, False)
        x_217 = None
        mul_16 = dropout_56 * 0.5
        dropout_56 = None
        residual_31 = residual_30 + mul_16
        residual_30 = mul_16 = None
        x_218 = torch.nn.functional.layer_norm(
            residual_31,
            (176,),
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_31 = (
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_ = None
        x_219 = torch.nn.functional.layer_norm(
            x_218,
            (176,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_220 = torch._C._nn.linear(
            x_219,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_219 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_221 = torch.nn.functional.silu(x_220, inplace=False)
        x_220 = None
        x_222 = torch.nn.functional.dropout(x_221, 0.1, False, False)
        x_221 = None
        x_223 = torch._C._nn.linear(
            x_222,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_222 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_58 = torch.nn.functional.dropout(x_223, 0.1, False, False)
        x_223 = None
        mul_17 = dropout_58 * 0.5
        dropout_58 = None
        residual_32 = x_218 + mul_17
        x_218 = mul_17 = None
        x_224 = torch.nn.functional.layer_norm(
            residual_32,
            (176,),
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_8_modules_norm_self_att_parameters_bias_
        ) = None
        linear_75 = torch._C._nn.linear(
            x_224,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_24 = linear_75.view(1, -1, 4, 44)
        linear_75 = None
        linear_76 = torch._C._nn.linear(
            x_224,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_16 = linear_76.view(1, -1, 4, 44)
        linear_76 = None
        linear_77 = torch._C._nn.linear(
            x_224,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_224 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_16 = linear_77.view(1, -1, 4, 44)
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
        p_16 = linear_78.view(1, -1, 4, 44)
        linear_78 = None
        p_17 = p_16.transpose(1, 2)
        p_16 = None
        add_61 = (
            q_26
            + l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_8 = add_61.transpose(1, 2)
        add_61 = None
        add_62 = (
            q_26
            + l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_
        )
        q_26 = (
            l_instance_modules_layers_modules_8_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_8 = add_62.transpose(1, 2)
        add_62 = None
        transpose_106 = p_17.transpose(-2, -1)
        p_17 = None
        matrix_bd_16 = torch.matmul(q_with_bias_v_8, transpose_106)
        q_with_bias_v_8 = transpose_106 = None
        x_225 = torch._C._nn.pad(matrix_bd_16, (1, 0), "constant", None)
        matrix_bd_16 = None
        x_226 = x_225.view(1, 4, -1, 131)
        x_225 = None
        getitem_18 = x_226[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_226 = None
        x_227 = getitem_18.view(1, 4, 131, 261)
        getitem_18 = None
        transpose_107 = k_17.transpose(-2, -1)
        k_17 = None
        matrix_ac_8 = torch.matmul(q_with_bias_u_8, transpose_107)
        q_with_bias_u_8 = transpose_107 = None
        matrix_bd_17 = x_227[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_227 = None
        add_63 = matrix_ac_8 + matrix_bd_17
        matrix_ac_8 = matrix_bd_17 = None
        scores_16 = add_63 / 6.6332495807108
        add_63 = None
        mask_8 = att_mask_3.unsqueeze(1)
        scores_17 = scores_16.masked_fill(mask_8, -10000.0)
        scores_16 = None
        softmax_8 = torch.softmax(scores_17, dim=-1)
        scores_17 = None
        attn_8 = softmax_8.masked_fill(mask_8, 0.0)
        softmax_8 = mask_8 = None
        p_attn_8 = torch.nn.functional.dropout(attn_8, 0.1, False, False)
        attn_8 = None
        x_228 = torch.matmul(p_attn_8, v_17)
        p_attn_8 = v_17 = None
        transpose_108 = x_228.transpose(1, 2)
        x_228 = None
        x_229 = transpose_108.reshape(1, -1, 176)
        transpose_108 = None
        out_8 = torch._C._nn.linear(
            x_229,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_229 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_60 = torch.nn.functional.dropout(out_8, 0.1, False, False)
        out_8 = None
        residual_33 = residual_32 + dropout_60
        residual_32 = dropout_60 = None
        x_230 = torch.nn.functional.layer_norm(
            residual_33,
            (176,),
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_
        ) = None
        x_231 = x_230.transpose(1, 2)
        x_230 = None
        x_232 = torch.conv1d(
            x_231,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_231 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_233 = torch.nn.functional.glu(x_232, dim=1)
        x_232 = None
        unsqueeze_20 = pad_mask_1.unsqueeze(1)
        x_234 = x_233.masked_fill(unsqueeze_20, 0.0)
        x_233 = unsqueeze_20 = None
        new_x_8 = torch._C._nn.pad(x_234, (15, 15), "constant", None)
        x_234 = None
        x_235 = torch.conv1d(
            new_x_8,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_8 = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_235 = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_237 = torch.nn.functional.silu(x_236, inplace=False)
        x_236 = None
        x_238 = torch.conv1d(
            x_237,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_237 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_239 = x_238.transpose(1, 2)
        x_238 = None
        dropout_61 = torch.nn.functional.dropout(x_239, 0.1, False, False)
        x_239 = None
        residual_34 = residual_33 + dropout_61
        residual_33 = dropout_61 = None
        x_240 = torch.nn.functional.layer_norm(
            residual_34,
            (176,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_241 = torch._C._nn.linear(
            x_240,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_240 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_242 = torch.nn.functional.silu(x_241, inplace=False)
        x_241 = None
        x_243 = torch.nn.functional.dropout(x_242, 0.1, False, False)
        x_242 = None
        x_244 = torch._C._nn.linear(
            x_243,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_243 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_63 = torch.nn.functional.dropout(x_244, 0.1, False, False)
        x_244 = None
        mul_18 = dropout_63 * 0.5
        dropout_63 = None
        residual_35 = residual_34 + mul_18
        residual_34 = mul_18 = None
        x_245 = torch.nn.functional.layer_norm(
            residual_35,
            (176,),
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_35 = (
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_ = None
        x_246 = torch.nn.functional.layer_norm(
            x_245,
            (176,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_247 = torch._C._nn.linear(
            x_246,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_246 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_248 = torch.nn.functional.silu(x_247, inplace=False)
        x_247 = None
        x_249 = torch.nn.functional.dropout(x_248, 0.1, False, False)
        x_248 = None
        x_250 = torch._C._nn.linear(
            x_249,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_249 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_65 = torch.nn.functional.dropout(x_250, 0.1, False, False)
        x_250 = None
        mul_19 = dropout_65 * 0.5
        dropout_65 = None
        residual_36 = x_245 + mul_19
        x_245 = mul_19 = None
        x_251 = torch.nn.functional.layer_norm(
            residual_36,
            (176,),
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_9_modules_norm_self_att_parameters_bias_
        ) = None
        linear_84 = torch._C._nn.linear(
            x_251,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_27 = linear_84.view(1, -1, 4, 44)
        linear_84 = None
        linear_85 = torch._C._nn.linear(
            x_251,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_18 = linear_85.view(1, -1, 4, 44)
        linear_85 = None
        linear_86 = torch._C._nn.linear(
            x_251,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_251 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_18 = linear_86.view(1, -1, 4, 44)
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
        p_18 = linear_87.view(1, -1, 4, 44)
        linear_87 = None
        p_19 = p_18.transpose(1, 2)
        p_18 = None
        add_68 = (
            q_29
            + l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_9 = add_68.transpose(1, 2)
        add_68 = None
        add_69 = (
            q_29
            + l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_
        )
        q_29 = (
            l_instance_modules_layers_modules_9_modules_self_attn_parameters_pos_bias_v_
        ) = None
        q_with_bias_v_9 = add_69.transpose(1, 2)
        add_69 = None
        transpose_118 = p_19.transpose(-2, -1)
        p_19 = None
        matrix_bd_18 = torch.matmul(q_with_bias_v_9, transpose_118)
        q_with_bias_v_9 = transpose_118 = None
        x_252 = torch._C._nn.pad(matrix_bd_18, (1, 0), "constant", None)
        matrix_bd_18 = None
        x_253 = x_252.view(1, 4, -1, 131)
        x_252 = None
        getitem_20 = x_253[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_253 = None
        x_254 = getitem_20.view(1, 4, 131, 261)
        getitem_20 = None
        transpose_119 = k_19.transpose(-2, -1)
        k_19 = None
        matrix_ac_9 = torch.matmul(q_with_bias_u_9, transpose_119)
        q_with_bias_u_9 = transpose_119 = None
        matrix_bd_19 = x_254[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_254 = None
        add_70 = matrix_ac_9 + matrix_bd_19
        matrix_ac_9 = matrix_bd_19 = None
        scores_18 = add_70 / 6.6332495807108
        add_70 = None
        mask_9 = att_mask_3.unsqueeze(1)
        scores_19 = scores_18.masked_fill(mask_9, -10000.0)
        scores_18 = None
        softmax_9 = torch.softmax(scores_19, dim=-1)
        scores_19 = None
        attn_9 = softmax_9.masked_fill(mask_9, 0.0)
        softmax_9 = mask_9 = None
        p_attn_9 = torch.nn.functional.dropout(attn_9, 0.1, False, False)
        attn_9 = None
        x_255 = torch.matmul(p_attn_9, v_19)
        p_attn_9 = v_19 = None
        transpose_120 = x_255.transpose(1, 2)
        x_255 = None
        x_256 = transpose_120.reshape(1, -1, 176)
        transpose_120 = None
        out_9 = torch._C._nn.linear(
            x_256,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_256 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_67 = torch.nn.functional.dropout(out_9, 0.1, False, False)
        out_9 = None
        residual_37 = residual_36 + dropout_67
        residual_36 = dropout_67 = None
        x_257 = torch.nn.functional.layer_norm(
            residual_37,
            (176,),
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_
        ) = None
        x_258 = x_257.transpose(1, 2)
        x_257 = None
        x_259 = torch.conv1d(
            x_258,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_258 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_260 = torch.nn.functional.glu(x_259, dim=1)
        x_259 = None
        unsqueeze_22 = pad_mask_1.unsqueeze(1)
        x_261 = x_260.masked_fill(unsqueeze_22, 0.0)
        x_260 = unsqueeze_22 = None
        new_x_9 = torch._C._nn.pad(x_261, (15, 15), "constant", None)
        x_261 = None
        x_262 = torch.conv1d(
            new_x_9,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_9 = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_264 = torch.nn.functional.silu(x_263, inplace=False)
        x_263 = None
        x_265 = torch.conv1d(
            x_264,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_264 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_266 = x_265.transpose(1, 2)
        x_265 = None
        dropout_68 = torch.nn.functional.dropout(x_266, 0.1, False, False)
        x_266 = None
        residual_38 = residual_37 + dropout_68
        residual_37 = dropout_68 = None
        x_267 = torch.nn.functional.layer_norm(
            residual_38,
            (176,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_268 = torch._C._nn.linear(
            x_267,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_267 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_269 = torch.nn.functional.silu(x_268, inplace=False)
        x_268 = None
        x_270 = torch.nn.functional.dropout(x_269, 0.1, False, False)
        x_269 = None
        x_271 = torch._C._nn.linear(
            x_270,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_270 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_70 = torch.nn.functional.dropout(x_271, 0.1, False, False)
        x_271 = None
        mul_20 = dropout_70 * 0.5
        dropout_70 = None
        residual_39 = residual_38 + mul_20
        residual_38 = mul_20 = None
        x_272 = torch.nn.functional.layer_norm(
            residual_39,
            (176,),
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_39 = (
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_ = None
        x_273 = torch.nn.functional.layer_norm(
            x_272,
            (176,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_274 = torch._C._nn.linear(
            x_273,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_273 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_275 = torch.nn.functional.silu(x_274, inplace=False)
        x_274 = None
        x_276 = torch.nn.functional.dropout(x_275, 0.1, False, False)
        x_275 = None
        x_277 = torch._C._nn.linear(
            x_276,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_276 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_72 = torch.nn.functional.dropout(x_277, 0.1, False, False)
        x_277 = None
        mul_21 = dropout_72 * 0.5
        dropout_72 = None
        residual_40 = x_272 + mul_21
        x_272 = mul_21 = None
        x_278 = torch.nn.functional.layer_norm(
            residual_40,
            (176,),
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_10_modules_norm_self_att_parameters_bias_
        ) = None
        linear_93 = torch._C._nn.linear(
            x_278,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_30 = linear_93.view(1, -1, 4, 44)
        linear_93 = None
        linear_94 = torch._C._nn.linear(
            x_278,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_20 = linear_94.view(1, -1, 4, 44)
        linear_94 = None
        linear_95 = torch._C._nn.linear(
            x_278,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_278 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_20 = linear_95.view(1, -1, 4, 44)
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
        p_20 = linear_96.view(1, -1, 4, 44)
        linear_96 = None
        p_21 = p_20.transpose(1, 2)
        p_20 = None
        add_75 = (
            q_32
            + l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_10 = add_75.transpose(1, 2)
        add_75 = None
        add_76 = (
            q_32
            + l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_
        )
        q_32 = l_instance_modules_layers_modules_10_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_10 = add_76.transpose(1, 2)
        add_76 = None
        transpose_130 = p_21.transpose(-2, -1)
        p_21 = None
        matrix_bd_20 = torch.matmul(q_with_bias_v_10, transpose_130)
        q_with_bias_v_10 = transpose_130 = None
        x_279 = torch._C._nn.pad(matrix_bd_20, (1, 0), "constant", None)
        matrix_bd_20 = None
        x_280 = x_279.view(1, 4, -1, 131)
        x_279 = None
        getitem_22 = x_280[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_280 = None
        x_281 = getitem_22.view(1, 4, 131, 261)
        getitem_22 = None
        transpose_131 = k_21.transpose(-2, -1)
        k_21 = None
        matrix_ac_10 = torch.matmul(q_with_bias_u_10, transpose_131)
        q_with_bias_u_10 = transpose_131 = None
        matrix_bd_21 = x_281[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_281 = None
        add_77 = matrix_ac_10 + matrix_bd_21
        matrix_ac_10 = matrix_bd_21 = None
        scores_20 = add_77 / 6.6332495807108
        add_77 = None
        mask_10 = att_mask_3.unsqueeze(1)
        scores_21 = scores_20.masked_fill(mask_10, -10000.0)
        scores_20 = None
        softmax_10 = torch.softmax(scores_21, dim=-1)
        scores_21 = None
        attn_10 = softmax_10.masked_fill(mask_10, 0.0)
        softmax_10 = mask_10 = None
        p_attn_10 = torch.nn.functional.dropout(attn_10, 0.1, False, False)
        attn_10 = None
        x_282 = torch.matmul(p_attn_10, v_21)
        p_attn_10 = v_21 = None
        transpose_132 = x_282.transpose(1, 2)
        x_282 = None
        x_283 = transpose_132.reshape(1, -1, 176)
        transpose_132 = None
        out_10 = torch._C._nn.linear(
            x_283,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_283 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_74 = torch.nn.functional.dropout(out_10, 0.1, False, False)
        out_10 = None
        residual_41 = residual_40 + dropout_74
        residual_40 = dropout_74 = None
        x_284 = torch.nn.functional.layer_norm(
            residual_41,
            (176,),
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_
        ) = None
        x_285 = x_284.transpose(1, 2)
        x_284 = None
        x_286 = torch.conv1d(
            x_285,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_285 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_287 = torch.nn.functional.glu(x_286, dim=1)
        x_286 = None
        unsqueeze_24 = pad_mask_1.unsqueeze(1)
        x_288 = x_287.masked_fill(unsqueeze_24, 0.0)
        x_287 = unsqueeze_24 = None
        new_x_10 = torch._C._nn.pad(x_288, (15, 15), "constant", None)
        x_288 = None
        x_289 = torch.conv1d(
            new_x_10,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_10 = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_290 = torch.nn.functional.batch_norm(
            x_289,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_289 = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_291 = torch.nn.functional.silu(x_290, inplace=False)
        x_290 = None
        x_292 = torch.conv1d(
            x_291,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_291 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_293 = x_292.transpose(1, 2)
        x_292 = None
        dropout_75 = torch.nn.functional.dropout(x_293, 0.1, False, False)
        x_293 = None
        residual_42 = residual_41 + dropout_75
        residual_41 = dropout_75 = None
        x_294 = torch.nn.functional.layer_norm(
            residual_42,
            (176,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_295 = torch._C._nn.linear(
            x_294,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_294 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_296 = torch.nn.functional.silu(x_295, inplace=False)
        x_295 = None
        x_297 = torch.nn.functional.dropout(x_296, 0.1, False, False)
        x_296 = None
        x_298 = torch._C._nn.linear(
            x_297,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_297 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_77 = torch.nn.functional.dropout(x_298, 0.1, False, False)
        x_298 = None
        mul_22 = dropout_77 * 0.5
        dropout_77 = None
        residual_43 = residual_42 + mul_22
        residual_42 = mul_22 = None
        x_299 = torch.nn.functional.layer_norm(
            residual_43,
            (176,),
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_43 = (
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_10_modules_norm_out_parameters_bias_
        ) = None
        x_300 = torch.nn.functional.layer_norm(
            x_299,
            (176,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_301 = torch._C._nn.linear(
            x_300,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_300 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_302 = torch.nn.functional.silu(x_301, inplace=False)
        x_301 = None
        x_303 = torch.nn.functional.dropout(x_302, 0.1, False, False)
        x_302 = None
        x_304 = torch._C._nn.linear(
            x_303,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_303 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_79 = torch.nn.functional.dropout(x_304, 0.1, False, False)
        x_304 = None
        mul_23 = dropout_79 * 0.5
        dropout_79 = None
        residual_44 = x_299 + mul_23
        x_299 = mul_23 = None
        x_305 = torch.nn.functional.layer_norm(
            residual_44,
            (176,),
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_11_modules_norm_self_att_parameters_bias_
        ) = None
        linear_102 = torch._C._nn.linear(
            x_305,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_33 = linear_102.view(1, -1, 4, 44)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            x_305,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_22 = linear_103.view(1, -1, 4, 44)
        linear_103 = None
        linear_104 = torch._C._nn.linear(
            x_305,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_305 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_22 = linear_104.view(1, -1, 4, 44)
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
        p_22 = linear_105.view(1, -1, 4, 44)
        linear_105 = None
        p_23 = p_22.transpose(1, 2)
        p_22 = None
        add_82 = (
            q_35
            + l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_11 = add_82.transpose(1, 2)
        add_82 = None
        add_83 = (
            q_35
            + l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_
        )
        q_35 = l_instance_modules_layers_modules_11_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_11 = add_83.transpose(1, 2)
        add_83 = None
        transpose_142 = p_23.transpose(-2, -1)
        p_23 = None
        matrix_bd_22 = torch.matmul(q_with_bias_v_11, transpose_142)
        q_with_bias_v_11 = transpose_142 = None
        x_306 = torch._C._nn.pad(matrix_bd_22, (1, 0), "constant", None)
        matrix_bd_22 = None
        x_307 = x_306.view(1, 4, -1, 131)
        x_306 = None
        getitem_24 = x_307[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_307 = None
        x_308 = getitem_24.view(1, 4, 131, 261)
        getitem_24 = None
        transpose_143 = k_23.transpose(-2, -1)
        k_23 = None
        matrix_ac_11 = torch.matmul(q_with_bias_u_11, transpose_143)
        q_with_bias_u_11 = transpose_143 = None
        matrix_bd_23 = x_308[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_308 = None
        add_84 = matrix_ac_11 + matrix_bd_23
        matrix_ac_11 = matrix_bd_23 = None
        scores_22 = add_84 / 6.6332495807108
        add_84 = None
        mask_11 = att_mask_3.unsqueeze(1)
        scores_23 = scores_22.masked_fill(mask_11, -10000.0)
        scores_22 = None
        softmax_11 = torch.softmax(scores_23, dim=-1)
        scores_23 = None
        attn_11 = softmax_11.masked_fill(mask_11, 0.0)
        softmax_11 = mask_11 = None
        p_attn_11 = torch.nn.functional.dropout(attn_11, 0.1, False, False)
        attn_11 = None
        x_309 = torch.matmul(p_attn_11, v_23)
        p_attn_11 = v_23 = None
        transpose_144 = x_309.transpose(1, 2)
        x_309 = None
        x_310 = transpose_144.reshape(1, -1, 176)
        transpose_144 = None
        out_11 = torch._C._nn.linear(
            x_310,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_310 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_81 = torch.nn.functional.dropout(out_11, 0.1, False, False)
        out_11 = None
        residual_45 = residual_44 + dropout_81
        residual_44 = dropout_81 = None
        x_311 = torch.nn.functional.layer_norm(
            residual_45,
            (176,),
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_
        ) = None
        x_312 = x_311.transpose(1, 2)
        x_311 = None
        x_313 = torch.conv1d(
            x_312,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_312 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_314 = torch.nn.functional.glu(x_313, dim=1)
        x_313 = None
        unsqueeze_26 = pad_mask_1.unsqueeze(1)
        x_315 = x_314.masked_fill(unsqueeze_26, 0.0)
        x_314 = unsqueeze_26 = None
        new_x_11 = torch._C._nn.pad(x_315, (15, 15), "constant", None)
        x_315 = None
        x_316 = torch.conv1d(
            new_x_11,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_11 = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_317 = torch.nn.functional.batch_norm(
            x_316,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_316 = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_318 = torch.nn.functional.silu(x_317, inplace=False)
        x_317 = None
        x_319 = torch.conv1d(
            x_318,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_318 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_320 = x_319.transpose(1, 2)
        x_319 = None
        dropout_82 = torch.nn.functional.dropout(x_320, 0.1, False, False)
        x_320 = None
        residual_46 = residual_45 + dropout_82
        residual_45 = dropout_82 = None
        x_321 = torch.nn.functional.layer_norm(
            residual_46,
            (176,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_322 = torch._C._nn.linear(
            x_321,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_321 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_323 = torch.nn.functional.silu(x_322, inplace=False)
        x_322 = None
        x_324 = torch.nn.functional.dropout(x_323, 0.1, False, False)
        x_323 = None
        x_325 = torch._C._nn.linear(
            x_324,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_324 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_84 = torch.nn.functional.dropout(x_325, 0.1, False, False)
        x_325 = None
        mul_24 = dropout_84 * 0.5
        dropout_84 = None
        residual_47 = residual_46 + mul_24
        residual_46 = mul_24 = None
        x_326 = torch.nn.functional.layer_norm(
            residual_47,
            (176,),
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_47 = (
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_11_modules_norm_out_parameters_bias_
        ) = None
        x_327 = torch.nn.functional.layer_norm(
            x_326,
            (176,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_328 = torch._C._nn.linear(
            x_327,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_327 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_329 = torch.nn.functional.silu(x_328, inplace=False)
        x_328 = None
        x_330 = torch.nn.functional.dropout(x_329, 0.1, False, False)
        x_329 = None
        x_331 = torch._C._nn.linear(
            x_330,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_330 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_86 = torch.nn.functional.dropout(x_331, 0.1, False, False)
        x_331 = None
        mul_25 = dropout_86 * 0.5
        dropout_86 = None
        residual_48 = x_326 + mul_25
        x_326 = mul_25 = None
        x_332 = torch.nn.functional.layer_norm(
            residual_48,
            (176,),
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_12_modules_norm_self_att_parameters_bias_
        ) = None
        linear_111 = torch._C._nn.linear(
            x_332,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_36 = linear_111.view(1, -1, 4, 44)
        linear_111 = None
        linear_112 = torch._C._nn.linear(
            x_332,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_24 = linear_112.view(1, -1, 4, 44)
        linear_112 = None
        linear_113 = torch._C._nn.linear(
            x_332,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_332 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_24 = linear_113.view(1, -1, 4, 44)
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
        p_24 = linear_114.view(1, -1, 4, 44)
        linear_114 = None
        p_25 = p_24.transpose(1, 2)
        p_24 = None
        add_89 = (
            q_38
            + l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_12 = add_89.transpose(1, 2)
        add_89 = None
        add_90 = (
            q_38
            + l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_
        )
        q_38 = l_instance_modules_layers_modules_12_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_12 = add_90.transpose(1, 2)
        add_90 = None
        transpose_154 = p_25.transpose(-2, -1)
        p_25 = None
        matrix_bd_24 = torch.matmul(q_with_bias_v_12, transpose_154)
        q_with_bias_v_12 = transpose_154 = None
        x_333 = torch._C._nn.pad(matrix_bd_24, (1, 0), "constant", None)
        matrix_bd_24 = None
        x_334 = x_333.view(1, 4, -1, 131)
        x_333 = None
        getitem_26 = x_334[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_334 = None
        x_335 = getitem_26.view(1, 4, 131, 261)
        getitem_26 = None
        transpose_155 = k_25.transpose(-2, -1)
        k_25 = None
        matrix_ac_12 = torch.matmul(q_with_bias_u_12, transpose_155)
        q_with_bias_u_12 = transpose_155 = None
        matrix_bd_25 = x_335[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_335 = None
        add_91 = matrix_ac_12 + matrix_bd_25
        matrix_ac_12 = matrix_bd_25 = None
        scores_24 = add_91 / 6.6332495807108
        add_91 = None
        mask_12 = att_mask_3.unsqueeze(1)
        scores_25 = scores_24.masked_fill(mask_12, -10000.0)
        scores_24 = None
        softmax_12 = torch.softmax(scores_25, dim=-1)
        scores_25 = None
        attn_12 = softmax_12.masked_fill(mask_12, 0.0)
        softmax_12 = mask_12 = None
        p_attn_12 = torch.nn.functional.dropout(attn_12, 0.1, False, False)
        attn_12 = None
        x_336 = torch.matmul(p_attn_12, v_25)
        p_attn_12 = v_25 = None
        transpose_156 = x_336.transpose(1, 2)
        x_336 = None
        x_337 = transpose_156.reshape(1, -1, 176)
        transpose_156 = None
        out_12 = torch._C._nn.linear(
            x_337,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_337 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_88 = torch.nn.functional.dropout(out_12, 0.1, False, False)
        out_12 = None
        residual_49 = residual_48 + dropout_88
        residual_48 = dropout_88 = None
        x_338 = torch.nn.functional.layer_norm(
            residual_49,
            (176,),
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_
        ) = None
        x_339 = x_338.transpose(1, 2)
        x_338 = None
        x_340 = torch.conv1d(
            x_339,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_339 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_341 = torch.nn.functional.glu(x_340, dim=1)
        x_340 = None
        unsqueeze_28 = pad_mask_1.unsqueeze(1)
        x_342 = x_341.masked_fill(unsqueeze_28, 0.0)
        x_341 = unsqueeze_28 = None
        new_x_12 = torch._C._nn.pad(x_342, (15, 15), "constant", None)
        x_342 = None
        x_343 = torch.conv1d(
            new_x_12,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_12 = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_344 = torch.nn.functional.batch_norm(
            x_343,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_343 = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_345 = torch.nn.functional.silu(x_344, inplace=False)
        x_344 = None
        x_346 = torch.conv1d(
            x_345,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_345 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_347 = x_346.transpose(1, 2)
        x_346 = None
        dropout_89 = torch.nn.functional.dropout(x_347, 0.1, False, False)
        x_347 = None
        residual_50 = residual_49 + dropout_89
        residual_49 = dropout_89 = None
        x_348 = torch.nn.functional.layer_norm(
            residual_50,
            (176,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_349 = torch._C._nn.linear(
            x_348,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_348 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_350 = torch.nn.functional.silu(x_349, inplace=False)
        x_349 = None
        x_351 = torch.nn.functional.dropout(x_350, 0.1, False, False)
        x_350 = None
        x_352 = torch._C._nn.linear(
            x_351,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_351 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_91 = torch.nn.functional.dropout(x_352, 0.1, False, False)
        x_352 = None
        mul_26 = dropout_91 * 0.5
        dropout_91 = None
        residual_51 = residual_50 + mul_26
        residual_50 = mul_26 = None
        x_353 = torch.nn.functional.layer_norm(
            residual_51,
            (176,),
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_51 = (
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_12_modules_norm_out_parameters_bias_
        ) = None
        x_354 = torch.nn.functional.layer_norm(
            x_353,
            (176,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_355 = torch._C._nn.linear(
            x_354,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_354 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_356 = torch.nn.functional.silu(x_355, inplace=False)
        x_355 = None
        x_357 = torch.nn.functional.dropout(x_356, 0.1, False, False)
        x_356 = None
        x_358 = torch._C._nn.linear(
            x_357,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_357 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_93 = torch.nn.functional.dropout(x_358, 0.1, False, False)
        x_358 = None
        mul_27 = dropout_93 * 0.5
        dropout_93 = None
        residual_52 = x_353 + mul_27
        x_353 = mul_27 = None
        x_359 = torch.nn.functional.layer_norm(
            residual_52,
            (176,),
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_13_modules_norm_self_att_parameters_bias_
        ) = None
        linear_120 = torch._C._nn.linear(
            x_359,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_39 = linear_120.view(1, -1, 4, 44)
        linear_120 = None
        linear_121 = torch._C._nn.linear(
            x_359,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_26 = linear_121.view(1, -1, 4, 44)
        linear_121 = None
        linear_122 = torch._C._nn.linear(
            x_359,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_359 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_26 = linear_122.view(1, -1, 4, 44)
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
        p_26 = linear_123.view(1, -1, 4, 44)
        linear_123 = None
        p_27 = p_26.transpose(1, 2)
        p_26 = None
        add_96 = (
            q_41
            + l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_13 = add_96.transpose(1, 2)
        add_96 = None
        add_97 = (
            q_41
            + l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_
        )
        q_41 = l_instance_modules_layers_modules_13_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_13 = add_97.transpose(1, 2)
        add_97 = None
        transpose_166 = p_27.transpose(-2, -1)
        p_27 = None
        matrix_bd_26 = torch.matmul(q_with_bias_v_13, transpose_166)
        q_with_bias_v_13 = transpose_166 = None
        x_360 = torch._C._nn.pad(matrix_bd_26, (1, 0), "constant", None)
        matrix_bd_26 = None
        x_361 = x_360.view(1, 4, -1, 131)
        x_360 = None
        getitem_28 = x_361[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_361 = None
        x_362 = getitem_28.view(1, 4, 131, 261)
        getitem_28 = None
        transpose_167 = k_27.transpose(-2, -1)
        k_27 = None
        matrix_ac_13 = torch.matmul(q_with_bias_u_13, transpose_167)
        q_with_bias_u_13 = transpose_167 = None
        matrix_bd_27 = x_362[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_362 = None
        add_98 = matrix_ac_13 + matrix_bd_27
        matrix_ac_13 = matrix_bd_27 = None
        scores_26 = add_98 / 6.6332495807108
        add_98 = None
        mask_13 = att_mask_3.unsqueeze(1)
        scores_27 = scores_26.masked_fill(mask_13, -10000.0)
        scores_26 = None
        softmax_13 = torch.softmax(scores_27, dim=-1)
        scores_27 = None
        attn_13 = softmax_13.masked_fill(mask_13, 0.0)
        softmax_13 = mask_13 = None
        p_attn_13 = torch.nn.functional.dropout(attn_13, 0.1, False, False)
        attn_13 = None
        x_363 = torch.matmul(p_attn_13, v_27)
        p_attn_13 = v_27 = None
        transpose_168 = x_363.transpose(1, 2)
        x_363 = None
        x_364 = transpose_168.reshape(1, -1, 176)
        transpose_168 = None
        out_13 = torch._C._nn.linear(
            x_364,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_364 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_95 = torch.nn.functional.dropout(out_13, 0.1, False, False)
        out_13 = None
        residual_53 = residual_52 + dropout_95
        residual_52 = dropout_95 = None
        x_365 = torch.nn.functional.layer_norm(
            residual_53,
            (176,),
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_
        ) = None
        x_366 = x_365.transpose(1, 2)
        x_365 = None
        x_367 = torch.conv1d(
            x_366,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_366 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_368 = torch.nn.functional.glu(x_367, dim=1)
        x_367 = None
        unsqueeze_30 = pad_mask_1.unsqueeze(1)
        x_369 = x_368.masked_fill(unsqueeze_30, 0.0)
        x_368 = unsqueeze_30 = None
        new_x_13 = torch._C._nn.pad(x_369, (15, 15), "constant", None)
        x_369 = None
        x_370 = torch.conv1d(
            new_x_13,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_13 = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_371 = torch.nn.functional.batch_norm(
            x_370,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_370 = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_372 = torch.nn.functional.silu(x_371, inplace=False)
        x_371 = None
        x_373 = torch.conv1d(
            x_372,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_372 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_374 = x_373.transpose(1, 2)
        x_373 = None
        dropout_96 = torch.nn.functional.dropout(x_374, 0.1, False, False)
        x_374 = None
        residual_54 = residual_53 + dropout_96
        residual_53 = dropout_96 = None
        x_375 = torch.nn.functional.layer_norm(
            residual_54,
            (176,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_376 = torch._C._nn.linear(
            x_375,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_375 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_377 = torch.nn.functional.silu(x_376, inplace=False)
        x_376 = None
        x_378 = torch.nn.functional.dropout(x_377, 0.1, False, False)
        x_377 = None
        x_379 = torch._C._nn.linear(
            x_378,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_378 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_98 = torch.nn.functional.dropout(x_379, 0.1, False, False)
        x_379 = None
        mul_28 = dropout_98 * 0.5
        dropout_98 = None
        residual_55 = residual_54 + mul_28
        residual_54 = mul_28 = None
        x_380 = torch.nn.functional.layer_norm(
            residual_55,
            (176,),
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_55 = (
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_13_modules_norm_out_parameters_bias_
        ) = None
        x_381 = torch.nn.functional.layer_norm(
            x_380,
            (176,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_382 = torch._C._nn.linear(
            x_381,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_381 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_383 = torch.nn.functional.silu(x_382, inplace=False)
        x_382 = None
        x_384 = torch.nn.functional.dropout(x_383, 0.1, False, False)
        x_383 = None
        x_385 = torch._C._nn.linear(
            x_384,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_384 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_100 = torch.nn.functional.dropout(x_385, 0.1, False, False)
        x_385 = None
        mul_29 = dropout_100 * 0.5
        dropout_100 = None
        residual_56 = x_380 + mul_29
        x_380 = mul_29 = None
        x_386 = torch.nn.functional.layer_norm(
            residual_56,
            (176,),
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_14_modules_norm_self_att_parameters_bias_
        ) = None
        linear_129 = torch._C._nn.linear(
            x_386,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_42 = linear_129.view(1, -1, 4, 44)
        linear_129 = None
        linear_130 = torch._C._nn.linear(
            x_386,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_28 = linear_130.view(1, -1, 4, 44)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            x_386,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_386 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_28 = linear_131.view(1, -1, 4, 44)
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
        p_28 = linear_132.view(1, -1, 4, 44)
        linear_132 = None
        p_29 = p_28.transpose(1, 2)
        p_28 = None
        add_103 = (
            q_44
            + l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_14 = add_103.transpose(1, 2)
        add_103 = None
        add_104 = (
            q_44
            + l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_
        )
        q_44 = l_instance_modules_layers_modules_14_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_14 = add_104.transpose(1, 2)
        add_104 = None
        transpose_178 = p_29.transpose(-2, -1)
        p_29 = None
        matrix_bd_28 = torch.matmul(q_with_bias_v_14, transpose_178)
        q_with_bias_v_14 = transpose_178 = None
        x_387 = torch._C._nn.pad(matrix_bd_28, (1, 0), "constant", None)
        matrix_bd_28 = None
        x_388 = x_387.view(1, 4, -1, 131)
        x_387 = None
        getitem_30 = x_388[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_388 = None
        x_389 = getitem_30.view(1, 4, 131, 261)
        getitem_30 = None
        transpose_179 = k_29.transpose(-2, -1)
        k_29 = None
        matrix_ac_14 = torch.matmul(q_with_bias_u_14, transpose_179)
        q_with_bias_u_14 = transpose_179 = None
        matrix_bd_29 = x_389[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_389 = None
        add_105 = matrix_ac_14 + matrix_bd_29
        matrix_ac_14 = matrix_bd_29 = None
        scores_28 = add_105 / 6.6332495807108
        add_105 = None
        mask_14 = att_mask_3.unsqueeze(1)
        scores_29 = scores_28.masked_fill(mask_14, -10000.0)
        scores_28 = None
        softmax_14 = torch.softmax(scores_29, dim=-1)
        scores_29 = None
        attn_14 = softmax_14.masked_fill(mask_14, 0.0)
        softmax_14 = mask_14 = None
        p_attn_14 = torch.nn.functional.dropout(attn_14, 0.1, False, False)
        attn_14 = None
        x_390 = torch.matmul(p_attn_14, v_29)
        p_attn_14 = v_29 = None
        transpose_180 = x_390.transpose(1, 2)
        x_390 = None
        x_391 = transpose_180.reshape(1, -1, 176)
        transpose_180 = None
        out_14 = torch._C._nn.linear(
            x_391,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_391 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_102 = torch.nn.functional.dropout(out_14, 0.1, False, False)
        out_14 = None
        residual_57 = residual_56 + dropout_102
        residual_56 = dropout_102 = None
        x_392 = torch.nn.functional.layer_norm(
            residual_57,
            (176,),
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_
        ) = None
        x_393 = x_392.transpose(1, 2)
        x_392 = None
        x_394 = torch.conv1d(
            x_393,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_393 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_395 = torch.nn.functional.glu(x_394, dim=1)
        x_394 = None
        unsqueeze_32 = pad_mask_1.unsqueeze(1)
        x_396 = x_395.masked_fill(unsqueeze_32, 0.0)
        x_395 = unsqueeze_32 = None
        new_x_14 = torch._C._nn.pad(x_396, (15, 15), "constant", None)
        x_396 = None
        x_397 = torch.conv1d(
            new_x_14,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_14 = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_398 = torch.nn.functional.batch_norm(
            x_397,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_397 = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_399 = torch.nn.functional.silu(x_398, inplace=False)
        x_398 = None
        x_400 = torch.conv1d(
            x_399,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_399 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_401 = x_400.transpose(1, 2)
        x_400 = None
        dropout_103 = torch.nn.functional.dropout(x_401, 0.1, False, False)
        x_401 = None
        residual_58 = residual_57 + dropout_103
        residual_57 = dropout_103 = None
        x_402 = torch.nn.functional.layer_norm(
            residual_58,
            (176,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_403 = torch._C._nn.linear(
            x_402,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_402 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_404 = torch.nn.functional.silu(x_403, inplace=False)
        x_403 = None
        x_405 = torch.nn.functional.dropout(x_404, 0.1, False, False)
        x_404 = None
        x_406 = torch._C._nn.linear(
            x_405,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_405 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_105 = torch.nn.functional.dropout(x_406, 0.1, False, False)
        x_406 = None
        mul_30 = dropout_105 * 0.5
        dropout_105 = None
        residual_59 = residual_58 + mul_30
        residual_58 = mul_30 = None
        x_407 = torch.nn.functional.layer_norm(
            residual_59,
            (176,),
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_59 = (
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_14_modules_norm_out_parameters_bias_
        ) = None
        x_408 = torch.nn.functional.layer_norm(
            x_407,
            (176,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_409 = torch._C._nn.linear(
            x_408,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_408 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_410 = torch.nn.functional.silu(x_409, inplace=False)
        x_409 = None
        x_411 = torch.nn.functional.dropout(x_410, 0.1, False, False)
        x_410 = None
        x_412 = torch._C._nn.linear(
            x_411,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_411 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_107 = torch.nn.functional.dropout(x_412, 0.1, False, False)
        x_412 = None
        mul_31 = dropout_107 * 0.5
        dropout_107 = None
        residual_60 = x_407 + mul_31
        x_407 = mul_31 = None
        x_413 = torch.nn.functional.layer_norm(
            residual_60,
            (176,),
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_weight_ = (
            l_instance_modules_layers_modules_15_modules_norm_self_att_parameters_bias_
        ) = None
        linear_138 = torch._C._nn.linear(
            x_413,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_45 = linear_138.view(1, -1, 4, 44)
        linear_138 = None
        linear_139 = torch._C._nn.linear(
            x_413,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_30 = linear_139.view(1, -1, 4, 44)
        linear_139 = None
        linear_140 = torch._C._nn.linear(
            x_413,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_413 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
        v_30 = linear_140.view(1, -1, 4, 44)
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
        pos_emb = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_pos_parameters_weight_ = (None)
        p_30 = linear_141.view(1, -1, 4, 44)
        linear_141 = None
        p_31 = p_30.transpose(1, 2)
        p_30 = None
        add_110 = (
            q_47
            + l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_
        )
        l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_u_ = (
            None
        )
        q_with_bias_u_15 = add_110.transpose(1, 2)
        add_110 = None
        add_111 = (
            q_47
            + l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_
        )
        q_47 = l_instance_modules_layers_modules_15_modules_self_attn_parameters_pos_bias_v_ = (None)
        q_with_bias_v_15 = add_111.transpose(1, 2)
        add_111 = None
        transpose_190 = p_31.transpose(-2, -1)
        p_31 = None
        matrix_bd_30 = torch.matmul(q_with_bias_v_15, transpose_190)
        q_with_bias_v_15 = transpose_190 = None
        x_414 = torch._C._nn.pad(matrix_bd_30, (1, 0), "constant", None)
        matrix_bd_30 = None
        x_415 = x_414.view(1, 4, -1, 131)
        x_414 = None
        getitem_32 = x_415[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_415 = None
        x_416 = getitem_32.view(1, 4, 131, 261)
        getitem_32 = None
        transpose_191 = k_31.transpose(-2, -1)
        k_31 = None
        matrix_ac_15 = torch.matmul(q_with_bias_u_15, transpose_191)
        q_with_bias_u_15 = transpose_191 = None
        matrix_bd_31 = x_416[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 131, None),
            )
        ]
        x_416 = None
        add_112 = matrix_ac_15 + matrix_bd_31
        matrix_ac_15 = matrix_bd_31 = None
        scores_30 = add_112 / 6.6332495807108
        add_112 = None
        mask_15 = att_mask_3.unsqueeze(1)
        att_mask_3 = None
        scores_31 = scores_30.masked_fill(mask_15, -10000.0)
        scores_30 = None
        softmax_15 = torch.softmax(scores_31, dim=-1)
        scores_31 = None
        attn_15 = softmax_15.masked_fill(mask_15, 0.0)
        softmax_15 = mask_15 = None
        p_attn_15 = torch.nn.functional.dropout(attn_15, 0.1, False, False)
        attn_15 = None
        x_417 = torch.matmul(p_attn_15, v_31)
        p_attn_15 = v_31 = None
        transpose_192 = x_417.transpose(1, 2)
        x_417 = None
        x_418 = transpose_192.reshape(1, -1, 176)
        transpose_192 = None
        out_15 = torch._C._nn.linear(
            x_418,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_418 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_109 = torch.nn.functional.dropout(out_15, 0.1, False, False)
        out_15 = None
        residual_61 = residual_60 + dropout_109
        residual_60 = dropout_109 = None
        x_419 = torch.nn.functional.layer_norm(
            residual_61,
            (176,),
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_
        ) = None
        x_420 = x_419.transpose(1, 2)
        x_419 = None
        x_421 = torch.conv1d(
            x_420,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_420 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_422 = torch.nn.functional.glu(x_421, dim=1)
        x_421 = None
        unsqueeze_34 = pad_mask_1.unsqueeze(1)
        pad_mask_1 = None
        x_423 = x_422.masked_fill(unsqueeze_34, 0.0)
        x_422 = unsqueeze_34 = None
        new_x_15 = torch._C._nn.pad(x_423, (15, 15), "constant", None)
        x_423 = None
        x_424 = torch.conv1d(
            new_x_15,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            176,
        )
        new_x_15 = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_425 = torch.nn.functional.batch_norm(
            x_424,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_424 = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_mean_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_buffers_running_var_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_426 = torch.nn.functional.silu(x_425, inplace=False)
        x_425 = None
        x_427 = torch.conv1d(
            x_426,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_426 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_428 = x_427.transpose(1, 2)
        x_427 = None
        dropout_110 = torch.nn.functional.dropout(x_428, 0.1, False, False)
        x_428 = None
        residual_62 = residual_61 + dropout_110
        residual_61 = dropout_110 = None
        x_429 = torch.nn.functional.layer_norm(
            residual_62,
            (176,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_430 = torch._C._nn.linear(
            x_429,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_429 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_431 = torch.nn.functional.silu(x_430, inplace=False)
        x_430 = None
        x_432 = torch.nn.functional.dropout(x_431, 0.1, False, False)
        x_431 = None
        x_433 = torch._C._nn.linear(
            x_432,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_432 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_112 = torch.nn.functional.dropout(x_433, 0.1, False, False)
        x_433 = None
        mul_32 = dropout_112 * 0.5
        dropout_112 = None
        residual_63 = residual_62 + mul_32
        residual_62 = mul_32 = None
        x_434 = torch.nn.functional.layer_norm(
            residual_63,
            (176,),
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_63 = (
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_weight_
        ) = (
            l_instance_modules_layers_modules_15_modules_norm_out_parameters_bias_
        ) = None
        audio_signal_2 = torch.transpose(x_434, 1, 2)
        x_434 = None
        length_1 = length.to(dtype=torch.int64)
        length = None
        return (audio_signal_2, length_1)
