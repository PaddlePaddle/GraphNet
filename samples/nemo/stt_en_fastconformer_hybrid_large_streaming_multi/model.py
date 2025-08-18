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
        add = to + 0
        to = None
        div = torch.div(add, 2)
        add = None
        lengths = div + 1.0
        div = None
        lengths_1 = torch.floor(lengths)
        lengths = None
        to_1 = lengths_1.to(dtype=torch.float32)
        lengths_1 = None
        add_2 = to_1 + 0
        to_1 = None
        div_1 = torch.div(add_2, 2)
        add_2 = None
        lengths_2 = div_1 + 1.0
        div_1 = None
        lengths_3 = torch.floor(lengths_2)
        lengths_2 = None
        to_2 = lengths_3.to(dtype=torch.float32)
        lengths_3 = None
        add_4 = to_2 + 0
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
        x_1 = torch._C._nn.pad(x, (2, 1, 2, 1), "constant", None)
        x = None
        x_2 = torch.conv2d(
            x_1,
            l_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_1 = (
            l_instance_modules_pre_encode_modules_conv_modules_0_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_0_parameters_bias_ = None
        input_1 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = None
        x_3 = torch._C._nn.pad(input_1, (2, 1, 2, 1), "constant", None)
        input_1 = None
        x_4 = torch.conv2d(
            x_3,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            256,
        )
        x_3 = (
            l_instance_modules_pre_encode_modules_conv_modules_2_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_2_parameters_bias_ = None
        input_2 = torch.conv2d(
            x_4,
            l_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = (
            l_instance_modules_pre_encode_modules_conv_modules_3_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_3_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        x_5 = torch._C._nn.pad(input_3, (2, 1, 2, 1), "constant", None)
        input_3 = None
        x_6 = torch.conv2d(
            x_5,
            l_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_,
            (2, 2),
            (0, 0),
            (1, 1),
            256,
        )
        x_5 = (
            l_instance_modules_pre_encode_modules_conv_modules_5_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_5_parameters_bias_ = None
        input_4 = torch.conv2d(
            x_6,
            l_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_,
            l_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_6 = (
            l_instance_modules_pre_encode_modules_conv_modules_6_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_conv_modules_6_parameters_bias_ = None
        input_5 = torch.nn.functional.relu(input_4, inplace=True)
        input_4 = None
        transpose_1 = input_5.transpose(1, 2)
        input_5 = None
        reshape = transpose_1.reshape(1, 66, -1)
        transpose_1 = None
        x_7 = torch._C._nn.linear(
            reshape,
            l_instance_modules_pre_encode_modules_out_parameters_weight_,
            l_instance_modules_pre_encode_modules_out_parameters_bias_,
        )
        reshape = (
            l_instance_modules_pre_encode_modules_out_parameters_weight_
        ) = l_instance_modules_pre_encode_modules_out_parameters_bias_ = None
        length = lengths_6.to(torch.int64)
        lengths_6 = None
        x_8 = x_7 * 22.627416997969522
        x_7 = None
        pos_emb = l_instance_modules_pos_enc_buffers_pe_[
            (slice(None, None, None), slice(4934, 5065, None))
        ]
        l_instance_modules_pos_enc_buffers_pe_ = None
        audio_signal_1 = torch.nn.functional.dropout(x_8, 0.1, False, False)
        x_8 = None
        att_mask = torch.ones(
            1, 66, 66, dtype=torch.bool, device=device(type="cuda", index=0)
        )
        chunk_idx = torch.arange(
            0, 66, dtype=torch.int32, device=device(type="cuda", index=0)
        )
        chunk_idx_1 = torch.div(chunk_idx, 14, rounding_mode="trunc")
        chunk_idx = None
        unsqueeze_1 = chunk_idx_1.unsqueeze(1)
        unsqueeze_2 = chunk_idx_1.unsqueeze(0)
        chunk_idx_1 = None
        diff_chunks = unsqueeze_1 - unsqueeze_2
        unsqueeze_1 = unsqueeze_2 = None
        le = torch.le(diff_chunks, 5)
        ge = torch.ge(diff_chunks, 0)
        diff_chunks = None
        chunked_limited_mask = torch.logical_and(le, ge)
        le = ge = None
        unsqueeze_3 = chunked_limited_mask.unsqueeze(0)
        chunked_limited_mask = None
        att_mask_1 = torch.logical_and(att_mask, unsqueeze_3)
        att_mask = unsqueeze_3 = None
        arange_1 = torch.arange(0, 66, device=device(type="cuda", index=0))
        expand = arange_1.expand(1, -1)
        arange_1 = None
        unsqueeze_4 = length.unsqueeze(-1)
        pad_mask = expand < unsqueeze_4
        expand = unsqueeze_4 = None
        unsqueeze_5 = pad_mask.unsqueeze(1)
        pad_mask_for_att_mask = unsqueeze_5.repeat([1, 66, 1])
        unsqueeze_5 = None
        transpose_2 = pad_mask_for_att_mask.transpose(1, 2)
        pad_mask_for_att_mask_1 = torch.logical_and(pad_mask_for_att_mask, transpose_2)
        pad_mask_for_att_mask = transpose_2 = None
        att_mask_2 = att_mask_1[
            (slice(None, None, None), slice(None, 66, None), slice(None, 66, None))
        ]
        att_mask_1 = None
        to_5 = att_mask_2.to(device(type="cuda", index=0))
        att_mask_2 = None
        att_mask_3 = torch.logical_and(pad_mask_for_att_mask_1, to_5)
        pad_mask_for_att_mask_1 = to_5 = None
        att_mask_4 = ~att_mask_3
        att_mask_3 = None
        pad_mask_1 = ~pad_mask
        pad_mask = None
        x_9 = torch.nn.functional.layer_norm(
            audio_signal_1,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_10 = torch._C._nn.linear(
            x_9,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_9 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_11 = torch.nn.functional.silu(x_10, inplace=False)
        x_10 = None
        x_12 = torch.nn.functional.dropout(x_11, 0.1, False, False)
        x_11 = None
        x_13 = torch._C._nn.linear(
            x_12,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_12 = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_2 = torch.nn.functional.dropout(x_13, 0.1, False, False)
        x_13 = None
        mul_1 = dropout_2 * 0.5
        dropout_2 = None
        residual = audio_signal_1 + mul_1
        audio_signal_1 = mul_1 = None
        x_14 = torch.nn.functional.layer_norm(
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
            x_14,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q = linear_3.view(1, -1, 8, 64)
        linear_3 = None
        linear_4 = torch._C._nn.linear(
            x_14,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k = linear_4.view(1, -1, 8, 64)
        linear_4 = None
        linear_5 = torch._C._nn.linear(
            x_14,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_14 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        x_15 = torch._C._nn.pad(matrix_bd, (1, 0), "constant", None)
        matrix_bd = None
        x_16 = x_15.view(1, 8, -1, 66)
        x_15 = None
        getitem_2 = x_16[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_16 = None
        x_17 = getitem_2.view(1, 8, 66, 131)
        getitem_2 = None
        transpose_11 = k_1.transpose(-2, -1)
        k_1 = None
        matrix_ac = torch.matmul(q_with_bias_u, transpose_11)
        q_with_bias_u = transpose_11 = None
        matrix_bd_1 = x_17[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_17 = None
        add_9 = matrix_ac + matrix_bd_1
        matrix_ac = matrix_bd_1 = None
        scores = add_9 / 8.0
        add_9 = None
        mask = att_mask_4.unsqueeze(1)
        scores_1 = scores.masked_fill(mask, -10000.0)
        scores = None
        softmax = torch.softmax(scores_1, dim=-1)
        scores_1 = None
        attn = softmax.masked_fill(mask, 0.0)
        softmax = mask = None
        p_attn = torch.nn.functional.dropout(attn, 0.1, False, False)
        attn = None
        x_18 = torch.matmul(p_attn, v_1)
        p_attn = v_1 = None
        transpose_12 = x_18.transpose(1, 2)
        x_18 = None
        x_19 = transpose_12.reshape(1, -1, 512)
        transpose_12 = None
        out = torch._C._nn.linear(
            x_19,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_19 = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_0_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_4 = torch.nn.functional.dropout(out, 0.1, False, False)
        out = None
        residual_1 = residual + dropout_4
        residual = dropout_4 = None
        x_20 = torch.nn.functional.layer_norm(
            residual_1,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_0_modules_norm_conv_parameters_bias_
        ) = None
        x_21 = x_20.transpose(1, 2)
        x_20 = None
        x_22 = torch.conv1d(
            x_21,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_21 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_23 = torch.nn.functional.glu(x_22, dim=1)
        x_22 = None
        unsqueeze_7 = pad_mask_1.unsqueeze(1)
        x_24 = x_23.masked_fill(unsqueeze_7, 0.0)
        x_23 = unsqueeze_7 = None
        new_x = torch._C._nn.pad(x_24, (8, 0), "constant", None)
        x_24 = None
        x_25 = torch.conv1d(
            new_x,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_26 = x_25.transpose(1, 2)
        x_25 = None
        x_27 = torch.nn.functional.layer_norm(
            x_26,
            (512,),
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_26 = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_28 = x_27.transpose(1, 2)
        x_27 = None
        x_29 = torch.nn.functional.silu(x_28, inplace=False)
        x_28 = None
        x_30 = torch.conv1d(
            x_29,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_29 = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_31 = x_30.transpose(1, 2)
        x_30 = None
        dropout_5 = torch.nn.functional.dropout(x_31, 0.1, False, False)
        x_31 = None
        residual_2 = residual_1 + dropout_5
        residual_1 = dropout_5 = None
        x_32 = torch.nn.functional.layer_norm(
            residual_2,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_33 = torch._C._nn.linear(
            x_32,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_32 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_34 = torch.nn.functional.silu(x_33, inplace=False)
        x_33 = None
        x_35 = torch.nn.functional.dropout(x_34, 0.1, False, False)
        x_34 = None
        x_36 = torch._C._nn.linear(
            x_35,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_35 = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_0_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_7 = torch.nn.functional.dropout(x_36, 0.1, False, False)
        x_36 = None
        mul_2 = dropout_7 * 0.5
        dropout_7 = None
        residual_3 = residual_2 + mul_2
        residual_2 = mul_2 = None
        x_37 = torch.nn.functional.layer_norm(
            residual_3,
            (512,),
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_3 = (
            l_instance_modules_layers_modules_0_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_0_modules_norm_out_parameters_bias_ = None
        x_38 = torch.nn.functional.layer_norm(
            x_37,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_39 = torch._C._nn.linear(
            x_38,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_38 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_40 = torch.nn.functional.silu(x_39, inplace=False)
        x_39 = None
        x_41 = torch.nn.functional.dropout(x_40, 0.1, False, False)
        x_40 = None
        x_42 = torch._C._nn.linear(
            x_41,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_41 = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_9 = torch.nn.functional.dropout(x_42, 0.1, False, False)
        x_42 = None
        mul_3 = dropout_9 * 0.5
        dropout_9 = None
        residual_4 = x_37 + mul_3
        x_37 = mul_3 = None
        x_43 = torch.nn.functional.layer_norm(
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
            x_43,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_3 = linear_12.view(1, -1, 8, 64)
        linear_12 = None
        linear_13 = torch._C._nn.linear(
            x_43,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_2 = linear_13.view(1, -1, 8, 64)
        linear_13 = None
        linear_14 = torch._C._nn.linear(
            x_43,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_43 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_24 = p_3.transpose(-2, -1)
        p_3 = None
        matrix_bd_2 = torch.matmul(q_with_bias_v_1, transpose_24)
        q_with_bias_v_1 = transpose_24 = None
        x_44 = torch._C._nn.pad(matrix_bd_2, (1, 0), "constant", None)
        matrix_bd_2 = None
        x_45 = x_44.view(1, 8, -1, 66)
        x_44 = None
        getitem_4 = x_45[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_45 = None
        x_46 = getitem_4.view(1, 8, 66, 131)
        getitem_4 = None
        transpose_25 = k_3.transpose(-2, -1)
        k_3 = None
        matrix_ac_1 = torch.matmul(q_with_bias_u_1, transpose_25)
        q_with_bias_u_1 = transpose_25 = None
        matrix_bd_3 = x_46[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_46 = None
        add_16 = matrix_ac_1 + matrix_bd_3
        matrix_ac_1 = matrix_bd_3 = None
        scores_2 = add_16 / 8.0
        add_16 = None
        mask_1 = att_mask_4.unsqueeze(1)
        scores_3 = scores_2.masked_fill(mask_1, -10000.0)
        scores_2 = None
        softmax_1 = torch.softmax(scores_3, dim=-1)
        scores_3 = None
        attn_1 = softmax_1.masked_fill(mask_1, 0.0)
        softmax_1 = mask_1 = None
        p_attn_1 = torch.nn.functional.dropout(attn_1, 0.1, False, False)
        attn_1 = None
        x_47 = torch.matmul(p_attn_1, v_3)
        p_attn_1 = v_3 = None
        transpose_26 = x_47.transpose(1, 2)
        x_47 = None
        x_48 = transpose_26.reshape(1, -1, 512)
        transpose_26 = None
        out_1 = torch._C._nn.linear(
            x_48,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_48 = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_1_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_11 = torch.nn.functional.dropout(out_1, 0.1, False, False)
        out_1 = None
        residual_5 = residual_4 + dropout_11
        residual_4 = dropout_11 = None
        x_49 = torch.nn.functional.layer_norm(
            residual_5,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_1_modules_norm_conv_parameters_bias_
        ) = None
        x_50 = x_49.transpose(1, 2)
        x_49 = None
        x_51 = torch.conv1d(
            x_50,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_50 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_52 = torch.nn.functional.glu(x_51, dim=1)
        x_51 = None
        unsqueeze_9 = pad_mask_1.unsqueeze(1)
        x_53 = x_52.masked_fill(unsqueeze_9, 0.0)
        x_52 = unsqueeze_9 = None
        new_x_1 = torch._C._nn.pad(x_53, (8, 0), "constant", None)
        x_53 = None
        x_54 = torch.conv1d(
            new_x_1,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_1 = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_55 = x_54.transpose(1, 2)
        x_54 = None
        x_56 = torch.nn.functional.layer_norm(
            x_55,
            (512,),
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_55 = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_57 = x_56.transpose(1, 2)
        x_56 = None
        x_58 = torch.nn.functional.silu(x_57, inplace=False)
        x_57 = None
        x_59 = torch.conv1d(
            x_58,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_58 = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_60 = x_59.transpose(1, 2)
        x_59 = None
        dropout_12 = torch.nn.functional.dropout(x_60, 0.1, False, False)
        x_60 = None
        residual_6 = residual_5 + dropout_12
        residual_5 = dropout_12 = None
        x_61 = torch.nn.functional.layer_norm(
            residual_6,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_62 = torch._C._nn.linear(
            x_61,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_61 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_63 = torch.nn.functional.silu(x_62, inplace=False)
        x_62 = None
        x_64 = torch.nn.functional.dropout(x_63, 0.1, False, False)
        x_63 = None
        x_65 = torch._C._nn.linear(
            x_64,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_64 = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_1_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_14 = torch.nn.functional.dropout(x_65, 0.1, False, False)
        x_65 = None
        mul_4 = dropout_14 * 0.5
        dropout_14 = None
        residual_7 = residual_6 + mul_4
        residual_6 = mul_4 = None
        x_66 = torch.nn.functional.layer_norm(
            residual_7,
            (512,),
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_7 = (
            l_instance_modules_layers_modules_1_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_1_modules_norm_out_parameters_bias_ = None
        x_67 = torch.nn.functional.layer_norm(
            x_66,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_68 = torch._C._nn.linear(
            x_67,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_67 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_69 = torch.nn.functional.silu(x_68, inplace=False)
        x_68 = None
        x_70 = torch.nn.functional.dropout(x_69, 0.1, False, False)
        x_69 = None
        x_71 = torch._C._nn.linear(
            x_70,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_70 = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_16 = torch.nn.functional.dropout(x_71, 0.1, False, False)
        x_71 = None
        mul_5 = dropout_16 * 0.5
        dropout_16 = None
        residual_8 = x_66 + mul_5
        x_66 = mul_5 = None
        x_72 = torch.nn.functional.layer_norm(
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
            x_72,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_6 = linear_21.view(1, -1, 8, 64)
        linear_21 = None
        linear_22 = torch._C._nn.linear(
            x_72,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_4 = linear_22.view(1, -1, 8, 64)
        linear_22 = None
        linear_23 = torch._C._nn.linear(
            x_72,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_72 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_38 = p_5.transpose(-2, -1)
        p_5 = None
        matrix_bd_4 = torch.matmul(q_with_bias_v_2, transpose_38)
        q_with_bias_v_2 = transpose_38 = None
        x_73 = torch._C._nn.pad(matrix_bd_4, (1, 0), "constant", None)
        matrix_bd_4 = None
        x_74 = x_73.view(1, 8, -1, 66)
        x_73 = None
        getitem_6 = x_74[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_74 = None
        x_75 = getitem_6.view(1, 8, 66, 131)
        getitem_6 = None
        transpose_39 = k_5.transpose(-2, -1)
        k_5 = None
        matrix_ac_2 = torch.matmul(q_with_bias_u_2, transpose_39)
        q_with_bias_u_2 = transpose_39 = None
        matrix_bd_5 = x_75[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_75 = None
        add_23 = matrix_ac_2 + matrix_bd_5
        matrix_ac_2 = matrix_bd_5 = None
        scores_4 = add_23 / 8.0
        add_23 = None
        mask_2 = att_mask_4.unsqueeze(1)
        scores_5 = scores_4.masked_fill(mask_2, -10000.0)
        scores_4 = None
        softmax_2 = torch.softmax(scores_5, dim=-1)
        scores_5 = None
        attn_2 = softmax_2.masked_fill(mask_2, 0.0)
        softmax_2 = mask_2 = None
        p_attn_2 = torch.nn.functional.dropout(attn_2, 0.1, False, False)
        attn_2 = None
        x_76 = torch.matmul(p_attn_2, v_5)
        p_attn_2 = v_5 = None
        transpose_40 = x_76.transpose(1, 2)
        x_76 = None
        x_77 = transpose_40.reshape(1, -1, 512)
        transpose_40 = None
        out_2 = torch._C._nn.linear(
            x_77,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_77 = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_2_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_18 = torch.nn.functional.dropout(out_2, 0.1, False, False)
        out_2 = None
        residual_9 = residual_8 + dropout_18
        residual_8 = dropout_18 = None
        x_78 = torch.nn.functional.layer_norm(
            residual_9,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_2_modules_norm_conv_parameters_bias_
        ) = None
        x_79 = x_78.transpose(1, 2)
        x_78 = None
        x_80 = torch.conv1d(
            x_79,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_79 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_81 = torch.nn.functional.glu(x_80, dim=1)
        x_80 = None
        unsqueeze_11 = pad_mask_1.unsqueeze(1)
        x_82 = x_81.masked_fill(unsqueeze_11, 0.0)
        x_81 = unsqueeze_11 = None
        new_x_2 = torch._C._nn.pad(x_82, (8, 0), "constant", None)
        x_82 = None
        x_83 = torch.conv1d(
            new_x_2,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_2 = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_84 = x_83.transpose(1, 2)
        x_83 = None
        x_85 = torch.nn.functional.layer_norm(
            x_84,
            (512,),
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_84 = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_86 = x_85.transpose(1, 2)
        x_85 = None
        x_87 = torch.nn.functional.silu(x_86, inplace=False)
        x_86 = None
        x_88 = torch.conv1d(
            x_87,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_87 = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_89 = x_88.transpose(1, 2)
        x_88 = None
        dropout_19 = torch.nn.functional.dropout(x_89, 0.1, False, False)
        x_89 = None
        residual_10 = residual_9 + dropout_19
        residual_9 = dropout_19 = None
        x_90 = torch.nn.functional.layer_norm(
            residual_10,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_91 = torch._C._nn.linear(
            x_90,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_90 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_92 = torch.nn.functional.silu(x_91, inplace=False)
        x_91 = None
        x_93 = torch.nn.functional.dropout(x_92, 0.1, False, False)
        x_92 = None
        x_94 = torch._C._nn.linear(
            x_93,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_93 = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_2_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_21 = torch.nn.functional.dropout(x_94, 0.1, False, False)
        x_94 = None
        mul_6 = dropout_21 * 0.5
        dropout_21 = None
        residual_11 = residual_10 + mul_6
        residual_10 = mul_6 = None
        x_95 = torch.nn.functional.layer_norm(
            residual_11,
            (512,),
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_11 = (
            l_instance_modules_layers_modules_2_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_2_modules_norm_out_parameters_bias_ = None
        x_96 = torch.nn.functional.layer_norm(
            x_95,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_97 = torch._C._nn.linear(
            x_96,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_96 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_98 = torch.nn.functional.silu(x_97, inplace=False)
        x_97 = None
        x_99 = torch.nn.functional.dropout(x_98, 0.1, False, False)
        x_98 = None
        x_100 = torch._C._nn.linear(
            x_99,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_99 = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_23 = torch.nn.functional.dropout(x_100, 0.1, False, False)
        x_100 = None
        mul_7 = dropout_23 * 0.5
        dropout_23 = None
        residual_12 = x_95 + mul_7
        x_95 = mul_7 = None
        x_101 = torch.nn.functional.layer_norm(
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
            x_101,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_9 = linear_30.view(1, -1, 8, 64)
        linear_30 = None
        linear_31 = torch._C._nn.linear(
            x_101,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_6 = linear_31.view(1, -1, 8, 64)
        linear_31 = None
        linear_32 = torch._C._nn.linear(
            x_101,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_101 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_52 = p_7.transpose(-2, -1)
        p_7 = None
        matrix_bd_6 = torch.matmul(q_with_bias_v_3, transpose_52)
        q_with_bias_v_3 = transpose_52 = None
        x_102 = torch._C._nn.pad(matrix_bd_6, (1, 0), "constant", None)
        matrix_bd_6 = None
        x_103 = x_102.view(1, 8, -1, 66)
        x_102 = None
        getitem_8 = x_103[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_103 = None
        x_104 = getitem_8.view(1, 8, 66, 131)
        getitem_8 = None
        transpose_53 = k_7.transpose(-2, -1)
        k_7 = None
        matrix_ac_3 = torch.matmul(q_with_bias_u_3, transpose_53)
        q_with_bias_u_3 = transpose_53 = None
        matrix_bd_7 = x_104[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_104 = None
        add_30 = matrix_ac_3 + matrix_bd_7
        matrix_ac_3 = matrix_bd_7 = None
        scores_6 = add_30 / 8.0
        add_30 = None
        mask_3 = att_mask_4.unsqueeze(1)
        scores_7 = scores_6.masked_fill(mask_3, -10000.0)
        scores_6 = None
        softmax_3 = torch.softmax(scores_7, dim=-1)
        scores_7 = None
        attn_3 = softmax_3.masked_fill(mask_3, 0.0)
        softmax_3 = mask_3 = None
        p_attn_3 = torch.nn.functional.dropout(attn_3, 0.1, False, False)
        attn_3 = None
        x_105 = torch.matmul(p_attn_3, v_7)
        p_attn_3 = v_7 = None
        transpose_54 = x_105.transpose(1, 2)
        x_105 = None
        x_106 = transpose_54.reshape(1, -1, 512)
        transpose_54 = None
        out_3 = torch._C._nn.linear(
            x_106,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_106 = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_3_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_25 = torch.nn.functional.dropout(out_3, 0.1, False, False)
        out_3 = None
        residual_13 = residual_12 + dropout_25
        residual_12 = dropout_25 = None
        x_107 = torch.nn.functional.layer_norm(
            residual_13,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_3_modules_norm_conv_parameters_bias_
        ) = None
        x_108 = x_107.transpose(1, 2)
        x_107 = None
        x_109 = torch.conv1d(
            x_108,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_108 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_110 = torch.nn.functional.glu(x_109, dim=1)
        x_109 = None
        unsqueeze_13 = pad_mask_1.unsqueeze(1)
        x_111 = x_110.masked_fill(unsqueeze_13, 0.0)
        x_110 = unsqueeze_13 = None
        new_x_3 = torch._C._nn.pad(x_111, (8, 0), "constant", None)
        x_111 = None
        x_112 = torch.conv1d(
            new_x_3,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_3 = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_113 = x_112.transpose(1, 2)
        x_112 = None
        x_114 = torch.nn.functional.layer_norm(
            x_113,
            (512,),
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_113 = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_115 = x_114.transpose(1, 2)
        x_114 = None
        x_116 = torch.nn.functional.silu(x_115, inplace=False)
        x_115 = None
        x_117 = torch.conv1d(
            x_116,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_116 = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_118 = x_117.transpose(1, 2)
        x_117 = None
        dropout_26 = torch.nn.functional.dropout(x_118, 0.1, False, False)
        x_118 = None
        residual_14 = residual_13 + dropout_26
        residual_13 = dropout_26 = None
        x_119 = torch.nn.functional.layer_norm(
            residual_14,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_120 = torch._C._nn.linear(
            x_119,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_119 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_121 = torch.nn.functional.silu(x_120, inplace=False)
        x_120 = None
        x_122 = torch.nn.functional.dropout(x_121, 0.1, False, False)
        x_121 = None
        x_123 = torch._C._nn.linear(
            x_122,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_122 = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_3_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_28 = torch.nn.functional.dropout(x_123, 0.1, False, False)
        x_123 = None
        mul_8 = dropout_28 * 0.5
        dropout_28 = None
        residual_15 = residual_14 + mul_8
        residual_14 = mul_8 = None
        x_124 = torch.nn.functional.layer_norm(
            residual_15,
            (512,),
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_15 = (
            l_instance_modules_layers_modules_3_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_3_modules_norm_out_parameters_bias_ = None
        x_125 = torch.nn.functional.layer_norm(
            x_124,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_126 = torch._C._nn.linear(
            x_125,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_125 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_127 = torch.nn.functional.silu(x_126, inplace=False)
        x_126 = None
        x_128 = torch.nn.functional.dropout(x_127, 0.1, False, False)
        x_127 = None
        x_129 = torch._C._nn.linear(
            x_128,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_128 = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_30 = torch.nn.functional.dropout(x_129, 0.1, False, False)
        x_129 = None
        mul_9 = dropout_30 * 0.5
        dropout_30 = None
        residual_16 = x_124 + mul_9
        x_124 = mul_9 = None
        x_130 = torch.nn.functional.layer_norm(
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
            x_130,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_12 = linear_39.view(1, -1, 8, 64)
        linear_39 = None
        linear_40 = torch._C._nn.linear(
            x_130,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_8 = linear_40.view(1, -1, 8, 64)
        linear_40 = None
        linear_41 = torch._C._nn.linear(
            x_130,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_130 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_66 = p_9.transpose(-2, -1)
        p_9 = None
        matrix_bd_8 = torch.matmul(q_with_bias_v_4, transpose_66)
        q_with_bias_v_4 = transpose_66 = None
        x_131 = torch._C._nn.pad(matrix_bd_8, (1, 0), "constant", None)
        matrix_bd_8 = None
        x_132 = x_131.view(1, 8, -1, 66)
        x_131 = None
        getitem_10 = x_132[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_132 = None
        x_133 = getitem_10.view(1, 8, 66, 131)
        getitem_10 = None
        transpose_67 = k_9.transpose(-2, -1)
        k_9 = None
        matrix_ac_4 = torch.matmul(q_with_bias_u_4, transpose_67)
        q_with_bias_u_4 = transpose_67 = None
        matrix_bd_9 = x_133[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_133 = None
        add_37 = matrix_ac_4 + matrix_bd_9
        matrix_ac_4 = matrix_bd_9 = None
        scores_8 = add_37 / 8.0
        add_37 = None
        mask_4 = att_mask_4.unsqueeze(1)
        scores_9 = scores_8.masked_fill(mask_4, -10000.0)
        scores_8 = None
        softmax_4 = torch.softmax(scores_9, dim=-1)
        scores_9 = None
        attn_4 = softmax_4.masked_fill(mask_4, 0.0)
        softmax_4 = mask_4 = None
        p_attn_4 = torch.nn.functional.dropout(attn_4, 0.1, False, False)
        attn_4 = None
        x_134 = torch.matmul(p_attn_4, v_9)
        p_attn_4 = v_9 = None
        transpose_68 = x_134.transpose(1, 2)
        x_134 = None
        x_135 = transpose_68.reshape(1, -1, 512)
        transpose_68 = None
        out_4 = torch._C._nn.linear(
            x_135,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_135 = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_4_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_32 = torch.nn.functional.dropout(out_4, 0.1, False, False)
        out_4 = None
        residual_17 = residual_16 + dropout_32
        residual_16 = dropout_32 = None
        x_136 = torch.nn.functional.layer_norm(
            residual_17,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_4_modules_norm_conv_parameters_bias_
        ) = None
        x_137 = x_136.transpose(1, 2)
        x_136 = None
        x_138 = torch.conv1d(
            x_137,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_137 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_139 = torch.nn.functional.glu(x_138, dim=1)
        x_138 = None
        unsqueeze_15 = pad_mask_1.unsqueeze(1)
        x_140 = x_139.masked_fill(unsqueeze_15, 0.0)
        x_139 = unsqueeze_15 = None
        new_x_4 = torch._C._nn.pad(x_140, (8, 0), "constant", None)
        x_140 = None
        x_141 = torch.conv1d(
            new_x_4,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_4 = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_142 = x_141.transpose(1, 2)
        x_141 = None
        x_143 = torch.nn.functional.layer_norm(
            x_142,
            (512,),
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_142 = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_144 = x_143.transpose(1, 2)
        x_143 = None
        x_145 = torch.nn.functional.silu(x_144, inplace=False)
        x_144 = None
        x_146 = torch.conv1d(
            x_145,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_145 = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_147 = x_146.transpose(1, 2)
        x_146 = None
        dropout_33 = torch.nn.functional.dropout(x_147, 0.1, False, False)
        x_147 = None
        residual_18 = residual_17 + dropout_33
        residual_17 = dropout_33 = None
        x_148 = torch.nn.functional.layer_norm(
            residual_18,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_149 = torch._C._nn.linear(
            x_148,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_148 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_150 = torch.nn.functional.silu(x_149, inplace=False)
        x_149 = None
        x_151 = torch.nn.functional.dropout(x_150, 0.1, False, False)
        x_150 = None
        x_152 = torch._C._nn.linear(
            x_151,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_151 = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_4_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_35 = torch.nn.functional.dropout(x_152, 0.1, False, False)
        x_152 = None
        mul_10 = dropout_35 * 0.5
        dropout_35 = None
        residual_19 = residual_18 + mul_10
        residual_18 = mul_10 = None
        x_153 = torch.nn.functional.layer_norm(
            residual_19,
            (512,),
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_19 = (
            l_instance_modules_layers_modules_4_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_4_modules_norm_out_parameters_bias_ = None
        x_154 = torch.nn.functional.layer_norm(
            x_153,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_155 = torch._C._nn.linear(
            x_154,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_154 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_156 = torch.nn.functional.silu(x_155, inplace=False)
        x_155 = None
        x_157 = torch.nn.functional.dropout(x_156, 0.1, False, False)
        x_156 = None
        x_158 = torch._C._nn.linear(
            x_157,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_157 = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_37 = torch.nn.functional.dropout(x_158, 0.1, False, False)
        x_158 = None
        mul_11 = dropout_37 * 0.5
        dropout_37 = None
        residual_20 = x_153 + mul_11
        x_153 = mul_11 = None
        x_159 = torch.nn.functional.layer_norm(
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
            x_159,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_15 = linear_48.view(1, -1, 8, 64)
        linear_48 = None
        linear_49 = torch._C._nn.linear(
            x_159,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_10 = linear_49.view(1, -1, 8, 64)
        linear_49 = None
        linear_50 = torch._C._nn.linear(
            x_159,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_159 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_80 = p_11.transpose(-2, -1)
        p_11 = None
        matrix_bd_10 = torch.matmul(q_with_bias_v_5, transpose_80)
        q_with_bias_v_5 = transpose_80 = None
        x_160 = torch._C._nn.pad(matrix_bd_10, (1, 0), "constant", None)
        matrix_bd_10 = None
        x_161 = x_160.view(1, 8, -1, 66)
        x_160 = None
        getitem_12 = x_161[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_161 = None
        x_162 = getitem_12.view(1, 8, 66, 131)
        getitem_12 = None
        transpose_81 = k_11.transpose(-2, -1)
        k_11 = None
        matrix_ac_5 = torch.matmul(q_with_bias_u_5, transpose_81)
        q_with_bias_u_5 = transpose_81 = None
        matrix_bd_11 = x_162[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_162 = None
        add_44 = matrix_ac_5 + matrix_bd_11
        matrix_ac_5 = matrix_bd_11 = None
        scores_10 = add_44 / 8.0
        add_44 = None
        mask_5 = att_mask_4.unsqueeze(1)
        scores_11 = scores_10.masked_fill(mask_5, -10000.0)
        scores_10 = None
        softmax_5 = torch.softmax(scores_11, dim=-1)
        scores_11 = None
        attn_5 = softmax_5.masked_fill(mask_5, 0.0)
        softmax_5 = mask_5 = None
        p_attn_5 = torch.nn.functional.dropout(attn_5, 0.1, False, False)
        attn_5 = None
        x_163 = torch.matmul(p_attn_5, v_11)
        p_attn_5 = v_11 = None
        transpose_82 = x_163.transpose(1, 2)
        x_163 = None
        x_164 = transpose_82.reshape(1, -1, 512)
        transpose_82 = None
        out_5 = torch._C._nn.linear(
            x_164,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_164 = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_5_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_39 = torch.nn.functional.dropout(out_5, 0.1, False, False)
        out_5 = None
        residual_21 = residual_20 + dropout_39
        residual_20 = dropout_39 = None
        x_165 = torch.nn.functional.layer_norm(
            residual_21,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_5_modules_norm_conv_parameters_bias_
        ) = None
        x_166 = x_165.transpose(1, 2)
        x_165 = None
        x_167 = torch.conv1d(
            x_166,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_166 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_168 = torch.nn.functional.glu(x_167, dim=1)
        x_167 = None
        unsqueeze_17 = pad_mask_1.unsqueeze(1)
        x_169 = x_168.masked_fill(unsqueeze_17, 0.0)
        x_168 = unsqueeze_17 = None
        new_x_5 = torch._C._nn.pad(x_169, (8, 0), "constant", None)
        x_169 = None
        x_170 = torch.conv1d(
            new_x_5,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_5 = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_171 = x_170.transpose(1, 2)
        x_170 = None
        x_172 = torch.nn.functional.layer_norm(
            x_171,
            (512,),
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_171 = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_173 = x_172.transpose(1, 2)
        x_172 = None
        x_174 = torch.nn.functional.silu(x_173, inplace=False)
        x_173 = None
        x_175 = torch.conv1d(
            x_174,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_174 = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_176 = x_175.transpose(1, 2)
        x_175 = None
        dropout_40 = torch.nn.functional.dropout(x_176, 0.1, False, False)
        x_176 = None
        residual_22 = residual_21 + dropout_40
        residual_21 = dropout_40 = None
        x_177 = torch.nn.functional.layer_norm(
            residual_22,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_178 = torch._C._nn.linear(
            x_177,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_177 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_179 = torch.nn.functional.silu(x_178, inplace=False)
        x_178 = None
        x_180 = torch.nn.functional.dropout(x_179, 0.1, False, False)
        x_179 = None
        x_181 = torch._C._nn.linear(
            x_180,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_180 = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_5_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_42 = torch.nn.functional.dropout(x_181, 0.1, False, False)
        x_181 = None
        mul_12 = dropout_42 * 0.5
        dropout_42 = None
        residual_23 = residual_22 + mul_12
        residual_22 = mul_12 = None
        x_182 = torch.nn.functional.layer_norm(
            residual_23,
            (512,),
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_23 = (
            l_instance_modules_layers_modules_5_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_5_modules_norm_out_parameters_bias_ = None
        x_183 = torch.nn.functional.layer_norm(
            x_182,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_184 = torch._C._nn.linear(
            x_183,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_183 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_185 = torch.nn.functional.silu(x_184, inplace=False)
        x_184 = None
        x_186 = torch.nn.functional.dropout(x_185, 0.1, False, False)
        x_185 = None
        x_187 = torch._C._nn.linear(
            x_186,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_186 = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_44 = torch.nn.functional.dropout(x_187, 0.1, False, False)
        x_187 = None
        mul_13 = dropout_44 * 0.5
        dropout_44 = None
        residual_24 = x_182 + mul_13
        x_182 = mul_13 = None
        x_188 = torch.nn.functional.layer_norm(
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
            x_188,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_18 = linear_57.view(1, -1, 8, 64)
        linear_57 = None
        linear_58 = torch._C._nn.linear(
            x_188,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_12 = linear_58.view(1, -1, 8, 64)
        linear_58 = None
        linear_59 = torch._C._nn.linear(
            x_188,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_188 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_94 = p_13.transpose(-2, -1)
        p_13 = None
        matrix_bd_12 = torch.matmul(q_with_bias_v_6, transpose_94)
        q_with_bias_v_6 = transpose_94 = None
        x_189 = torch._C._nn.pad(matrix_bd_12, (1, 0), "constant", None)
        matrix_bd_12 = None
        x_190 = x_189.view(1, 8, -1, 66)
        x_189 = None
        getitem_14 = x_190[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_190 = None
        x_191 = getitem_14.view(1, 8, 66, 131)
        getitem_14 = None
        transpose_95 = k_13.transpose(-2, -1)
        k_13 = None
        matrix_ac_6 = torch.matmul(q_with_bias_u_6, transpose_95)
        q_with_bias_u_6 = transpose_95 = None
        matrix_bd_13 = x_191[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_191 = None
        add_51 = matrix_ac_6 + matrix_bd_13
        matrix_ac_6 = matrix_bd_13 = None
        scores_12 = add_51 / 8.0
        add_51 = None
        mask_6 = att_mask_4.unsqueeze(1)
        scores_13 = scores_12.masked_fill(mask_6, -10000.0)
        scores_12 = None
        softmax_6 = torch.softmax(scores_13, dim=-1)
        scores_13 = None
        attn_6 = softmax_6.masked_fill(mask_6, 0.0)
        softmax_6 = mask_6 = None
        p_attn_6 = torch.nn.functional.dropout(attn_6, 0.1, False, False)
        attn_6 = None
        x_192 = torch.matmul(p_attn_6, v_13)
        p_attn_6 = v_13 = None
        transpose_96 = x_192.transpose(1, 2)
        x_192 = None
        x_193 = transpose_96.reshape(1, -1, 512)
        transpose_96 = None
        out_6 = torch._C._nn.linear(
            x_193,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_193 = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_6_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_46 = torch.nn.functional.dropout(out_6, 0.1, False, False)
        out_6 = None
        residual_25 = residual_24 + dropout_46
        residual_24 = dropout_46 = None
        x_194 = torch.nn.functional.layer_norm(
            residual_25,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_6_modules_norm_conv_parameters_bias_
        ) = None
        x_195 = x_194.transpose(1, 2)
        x_194 = None
        x_196 = torch.conv1d(
            x_195,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_195 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_197 = torch.nn.functional.glu(x_196, dim=1)
        x_196 = None
        unsqueeze_19 = pad_mask_1.unsqueeze(1)
        x_198 = x_197.masked_fill(unsqueeze_19, 0.0)
        x_197 = unsqueeze_19 = None
        new_x_6 = torch._C._nn.pad(x_198, (8, 0), "constant", None)
        x_198 = None
        x_199 = torch.conv1d(
            new_x_6,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_6 = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_200 = x_199.transpose(1, 2)
        x_199 = None
        x_201 = torch.nn.functional.layer_norm(
            x_200,
            (512,),
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_200 = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_202 = x_201.transpose(1, 2)
        x_201 = None
        x_203 = torch.nn.functional.silu(x_202, inplace=False)
        x_202 = None
        x_204 = torch.conv1d(
            x_203,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_203 = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_205 = x_204.transpose(1, 2)
        x_204 = None
        dropout_47 = torch.nn.functional.dropout(x_205, 0.1, False, False)
        x_205 = None
        residual_26 = residual_25 + dropout_47
        residual_25 = dropout_47 = None
        x_206 = torch.nn.functional.layer_norm(
            residual_26,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_207 = torch._C._nn.linear(
            x_206,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_206 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_208 = torch.nn.functional.silu(x_207, inplace=False)
        x_207 = None
        x_209 = torch.nn.functional.dropout(x_208, 0.1, False, False)
        x_208 = None
        x_210 = torch._C._nn.linear(
            x_209,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_209 = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_6_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_49 = torch.nn.functional.dropout(x_210, 0.1, False, False)
        x_210 = None
        mul_14 = dropout_49 * 0.5
        dropout_49 = None
        residual_27 = residual_26 + mul_14
        residual_26 = mul_14 = None
        x_211 = torch.nn.functional.layer_norm(
            residual_27,
            (512,),
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_27 = (
            l_instance_modules_layers_modules_6_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_6_modules_norm_out_parameters_bias_ = None
        x_212 = torch.nn.functional.layer_norm(
            x_211,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_213 = torch._C._nn.linear(
            x_212,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_212 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_214 = torch.nn.functional.silu(x_213, inplace=False)
        x_213 = None
        x_215 = torch.nn.functional.dropout(x_214, 0.1, False, False)
        x_214 = None
        x_216 = torch._C._nn.linear(
            x_215,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_215 = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_51 = torch.nn.functional.dropout(x_216, 0.1, False, False)
        x_216 = None
        mul_15 = dropout_51 * 0.5
        dropout_51 = None
        residual_28 = x_211 + mul_15
        x_211 = mul_15 = None
        x_217 = torch.nn.functional.layer_norm(
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
            x_217,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_21 = linear_66.view(1, -1, 8, 64)
        linear_66 = None
        linear_67 = torch._C._nn.linear(
            x_217,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_14 = linear_67.view(1, -1, 8, 64)
        linear_67 = None
        linear_68 = torch._C._nn.linear(
            x_217,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_217 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_108 = p_15.transpose(-2, -1)
        p_15 = None
        matrix_bd_14 = torch.matmul(q_with_bias_v_7, transpose_108)
        q_with_bias_v_7 = transpose_108 = None
        x_218 = torch._C._nn.pad(matrix_bd_14, (1, 0), "constant", None)
        matrix_bd_14 = None
        x_219 = x_218.view(1, 8, -1, 66)
        x_218 = None
        getitem_16 = x_219[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_219 = None
        x_220 = getitem_16.view(1, 8, 66, 131)
        getitem_16 = None
        transpose_109 = k_15.transpose(-2, -1)
        k_15 = None
        matrix_ac_7 = torch.matmul(q_with_bias_u_7, transpose_109)
        q_with_bias_u_7 = transpose_109 = None
        matrix_bd_15 = x_220[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_220 = None
        add_58 = matrix_ac_7 + matrix_bd_15
        matrix_ac_7 = matrix_bd_15 = None
        scores_14 = add_58 / 8.0
        add_58 = None
        mask_7 = att_mask_4.unsqueeze(1)
        scores_15 = scores_14.masked_fill(mask_7, -10000.0)
        scores_14 = None
        softmax_7 = torch.softmax(scores_15, dim=-1)
        scores_15 = None
        attn_7 = softmax_7.masked_fill(mask_7, 0.0)
        softmax_7 = mask_7 = None
        p_attn_7 = torch.nn.functional.dropout(attn_7, 0.1, False, False)
        attn_7 = None
        x_221 = torch.matmul(p_attn_7, v_15)
        p_attn_7 = v_15 = None
        transpose_110 = x_221.transpose(1, 2)
        x_221 = None
        x_222 = transpose_110.reshape(1, -1, 512)
        transpose_110 = None
        out_7 = torch._C._nn.linear(
            x_222,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_222 = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_7_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_53 = torch.nn.functional.dropout(out_7, 0.1, False, False)
        out_7 = None
        residual_29 = residual_28 + dropout_53
        residual_28 = dropout_53 = None
        x_223 = torch.nn.functional.layer_norm(
            residual_29,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_7_modules_norm_conv_parameters_bias_
        ) = None
        x_224 = x_223.transpose(1, 2)
        x_223 = None
        x_225 = torch.conv1d(
            x_224,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_224 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_226 = torch.nn.functional.glu(x_225, dim=1)
        x_225 = None
        unsqueeze_21 = pad_mask_1.unsqueeze(1)
        x_227 = x_226.masked_fill(unsqueeze_21, 0.0)
        x_226 = unsqueeze_21 = None
        new_x_7 = torch._C._nn.pad(x_227, (8, 0), "constant", None)
        x_227 = None
        x_228 = torch.conv1d(
            new_x_7,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_7 = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_229 = x_228.transpose(1, 2)
        x_228 = None
        x_230 = torch.nn.functional.layer_norm(
            x_229,
            (512,),
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_229 = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_231 = x_230.transpose(1, 2)
        x_230 = None
        x_232 = torch.nn.functional.silu(x_231, inplace=False)
        x_231 = None
        x_233 = torch.conv1d(
            x_232,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_232 = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_234 = x_233.transpose(1, 2)
        x_233 = None
        dropout_54 = torch.nn.functional.dropout(x_234, 0.1, False, False)
        x_234 = None
        residual_30 = residual_29 + dropout_54
        residual_29 = dropout_54 = None
        x_235 = torch.nn.functional.layer_norm(
            residual_30,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_236 = torch._C._nn.linear(
            x_235,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_235 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_237 = torch.nn.functional.silu(x_236, inplace=False)
        x_236 = None
        x_238 = torch.nn.functional.dropout(x_237, 0.1, False, False)
        x_237 = None
        x_239 = torch._C._nn.linear(
            x_238,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_238 = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_7_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_56 = torch.nn.functional.dropout(x_239, 0.1, False, False)
        x_239 = None
        mul_16 = dropout_56 * 0.5
        dropout_56 = None
        residual_31 = residual_30 + mul_16
        residual_30 = mul_16 = None
        x_240 = torch.nn.functional.layer_norm(
            residual_31,
            (512,),
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_31 = (
            l_instance_modules_layers_modules_7_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_7_modules_norm_out_parameters_bias_ = None
        x_241 = torch.nn.functional.layer_norm(
            x_240,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_242 = torch._C._nn.linear(
            x_241,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_241 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_243 = torch.nn.functional.silu(x_242, inplace=False)
        x_242 = None
        x_244 = torch.nn.functional.dropout(x_243, 0.1, False, False)
        x_243 = None
        x_245 = torch._C._nn.linear(
            x_244,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_244 = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_58 = torch.nn.functional.dropout(x_245, 0.1, False, False)
        x_245 = None
        mul_17 = dropout_58 * 0.5
        dropout_58 = None
        residual_32 = x_240 + mul_17
        x_240 = mul_17 = None
        x_246 = torch.nn.functional.layer_norm(
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
            x_246,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_24 = linear_75.view(1, -1, 8, 64)
        linear_75 = None
        linear_76 = torch._C._nn.linear(
            x_246,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_16 = linear_76.view(1, -1, 8, 64)
        linear_76 = None
        linear_77 = torch._C._nn.linear(
            x_246,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_246 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_122 = p_17.transpose(-2, -1)
        p_17 = None
        matrix_bd_16 = torch.matmul(q_with_bias_v_8, transpose_122)
        q_with_bias_v_8 = transpose_122 = None
        x_247 = torch._C._nn.pad(matrix_bd_16, (1, 0), "constant", None)
        matrix_bd_16 = None
        x_248 = x_247.view(1, 8, -1, 66)
        x_247 = None
        getitem_18 = x_248[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_248 = None
        x_249 = getitem_18.view(1, 8, 66, 131)
        getitem_18 = None
        transpose_123 = k_17.transpose(-2, -1)
        k_17 = None
        matrix_ac_8 = torch.matmul(q_with_bias_u_8, transpose_123)
        q_with_bias_u_8 = transpose_123 = None
        matrix_bd_17 = x_249[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_249 = None
        add_65 = matrix_ac_8 + matrix_bd_17
        matrix_ac_8 = matrix_bd_17 = None
        scores_16 = add_65 / 8.0
        add_65 = None
        mask_8 = att_mask_4.unsqueeze(1)
        scores_17 = scores_16.masked_fill(mask_8, -10000.0)
        scores_16 = None
        softmax_8 = torch.softmax(scores_17, dim=-1)
        scores_17 = None
        attn_8 = softmax_8.masked_fill(mask_8, 0.0)
        softmax_8 = mask_8 = None
        p_attn_8 = torch.nn.functional.dropout(attn_8, 0.1, False, False)
        attn_8 = None
        x_250 = torch.matmul(p_attn_8, v_17)
        p_attn_8 = v_17 = None
        transpose_124 = x_250.transpose(1, 2)
        x_250 = None
        x_251 = transpose_124.reshape(1, -1, 512)
        transpose_124 = None
        out_8 = torch._C._nn.linear(
            x_251,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_251 = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_8_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_60 = torch.nn.functional.dropout(out_8, 0.1, False, False)
        out_8 = None
        residual_33 = residual_32 + dropout_60
        residual_32 = dropout_60 = None
        x_252 = torch.nn.functional.layer_norm(
            residual_33,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_8_modules_norm_conv_parameters_bias_
        ) = None
        x_253 = x_252.transpose(1, 2)
        x_252 = None
        x_254 = torch.conv1d(
            x_253,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_253 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_255 = torch.nn.functional.glu(x_254, dim=1)
        x_254 = None
        unsqueeze_23 = pad_mask_1.unsqueeze(1)
        x_256 = x_255.masked_fill(unsqueeze_23, 0.0)
        x_255 = unsqueeze_23 = None
        new_x_8 = torch._C._nn.pad(x_256, (8, 0), "constant", None)
        x_256 = None
        x_257 = torch.conv1d(
            new_x_8,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_8 = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_258 = x_257.transpose(1, 2)
        x_257 = None
        x_259 = torch.nn.functional.layer_norm(
            x_258,
            (512,),
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_258 = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_260 = x_259.transpose(1, 2)
        x_259 = None
        x_261 = torch.nn.functional.silu(x_260, inplace=False)
        x_260 = None
        x_262 = torch.conv1d(
            x_261,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_261 = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_263 = x_262.transpose(1, 2)
        x_262 = None
        dropout_61 = torch.nn.functional.dropout(x_263, 0.1, False, False)
        x_263 = None
        residual_34 = residual_33 + dropout_61
        residual_33 = dropout_61 = None
        x_264 = torch.nn.functional.layer_norm(
            residual_34,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_265 = torch._C._nn.linear(
            x_264,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_264 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_266 = torch.nn.functional.silu(x_265, inplace=False)
        x_265 = None
        x_267 = torch.nn.functional.dropout(x_266, 0.1, False, False)
        x_266 = None
        x_268 = torch._C._nn.linear(
            x_267,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_267 = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_8_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_63 = torch.nn.functional.dropout(x_268, 0.1, False, False)
        x_268 = None
        mul_18 = dropout_63 * 0.5
        dropout_63 = None
        residual_35 = residual_34 + mul_18
        residual_34 = mul_18 = None
        x_269 = torch.nn.functional.layer_norm(
            residual_35,
            (512,),
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_35 = (
            l_instance_modules_layers_modules_8_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_8_modules_norm_out_parameters_bias_ = None
        x_270 = torch.nn.functional.layer_norm(
            x_269,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_271 = torch._C._nn.linear(
            x_270,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_270 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_272 = torch.nn.functional.silu(x_271, inplace=False)
        x_271 = None
        x_273 = torch.nn.functional.dropout(x_272, 0.1, False, False)
        x_272 = None
        x_274 = torch._C._nn.linear(
            x_273,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_273 = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_65 = torch.nn.functional.dropout(x_274, 0.1, False, False)
        x_274 = None
        mul_19 = dropout_65 * 0.5
        dropout_65 = None
        residual_36 = x_269 + mul_19
        x_269 = mul_19 = None
        x_275 = torch.nn.functional.layer_norm(
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
            x_275,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_27 = linear_84.view(1, -1, 8, 64)
        linear_84 = None
        linear_85 = torch._C._nn.linear(
            x_275,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_18 = linear_85.view(1, -1, 8, 64)
        linear_85 = None
        linear_86 = torch._C._nn.linear(
            x_275,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_275 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_136 = p_19.transpose(-2, -1)
        p_19 = None
        matrix_bd_18 = torch.matmul(q_with_bias_v_9, transpose_136)
        q_with_bias_v_9 = transpose_136 = None
        x_276 = torch._C._nn.pad(matrix_bd_18, (1, 0), "constant", None)
        matrix_bd_18 = None
        x_277 = x_276.view(1, 8, -1, 66)
        x_276 = None
        getitem_20 = x_277[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_277 = None
        x_278 = getitem_20.view(1, 8, 66, 131)
        getitem_20 = None
        transpose_137 = k_19.transpose(-2, -1)
        k_19 = None
        matrix_ac_9 = torch.matmul(q_with_bias_u_9, transpose_137)
        q_with_bias_u_9 = transpose_137 = None
        matrix_bd_19 = x_278[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_278 = None
        add_72 = matrix_ac_9 + matrix_bd_19
        matrix_ac_9 = matrix_bd_19 = None
        scores_18 = add_72 / 8.0
        add_72 = None
        mask_9 = att_mask_4.unsqueeze(1)
        scores_19 = scores_18.masked_fill(mask_9, -10000.0)
        scores_18 = None
        softmax_9 = torch.softmax(scores_19, dim=-1)
        scores_19 = None
        attn_9 = softmax_9.masked_fill(mask_9, 0.0)
        softmax_9 = mask_9 = None
        p_attn_9 = torch.nn.functional.dropout(attn_9, 0.1, False, False)
        attn_9 = None
        x_279 = torch.matmul(p_attn_9, v_19)
        p_attn_9 = v_19 = None
        transpose_138 = x_279.transpose(1, 2)
        x_279 = None
        x_280 = transpose_138.reshape(1, -1, 512)
        transpose_138 = None
        out_9 = torch._C._nn.linear(
            x_280,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_280 = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_9_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_67 = torch.nn.functional.dropout(out_9, 0.1, False, False)
        out_9 = None
        residual_37 = residual_36 + dropout_67
        residual_36 = dropout_67 = None
        x_281 = torch.nn.functional.layer_norm(
            residual_37,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_9_modules_norm_conv_parameters_bias_
        ) = None
        x_282 = x_281.transpose(1, 2)
        x_281 = None
        x_283 = torch.conv1d(
            x_282,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_282 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_284 = torch.nn.functional.glu(x_283, dim=1)
        x_283 = None
        unsqueeze_25 = pad_mask_1.unsqueeze(1)
        x_285 = x_284.masked_fill(unsqueeze_25, 0.0)
        x_284 = unsqueeze_25 = None
        new_x_9 = torch._C._nn.pad(x_285, (8, 0), "constant", None)
        x_285 = None
        x_286 = torch.conv1d(
            new_x_9,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_9 = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_287 = x_286.transpose(1, 2)
        x_286 = None
        x_288 = torch.nn.functional.layer_norm(
            x_287,
            (512,),
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_287 = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_289 = x_288.transpose(1, 2)
        x_288 = None
        x_290 = torch.nn.functional.silu(x_289, inplace=False)
        x_289 = None
        x_291 = torch.conv1d(
            x_290,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_290 = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_292 = x_291.transpose(1, 2)
        x_291 = None
        dropout_68 = torch.nn.functional.dropout(x_292, 0.1, False, False)
        x_292 = None
        residual_38 = residual_37 + dropout_68
        residual_37 = dropout_68 = None
        x_293 = torch.nn.functional.layer_norm(
            residual_38,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_294 = torch._C._nn.linear(
            x_293,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_293 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_295 = torch.nn.functional.silu(x_294, inplace=False)
        x_294 = None
        x_296 = torch.nn.functional.dropout(x_295, 0.1, False, False)
        x_295 = None
        x_297 = torch._C._nn.linear(
            x_296,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_296 = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_9_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_70 = torch.nn.functional.dropout(x_297, 0.1, False, False)
        x_297 = None
        mul_20 = dropout_70 * 0.5
        dropout_70 = None
        residual_39 = residual_38 + mul_20
        residual_38 = mul_20 = None
        x_298 = torch.nn.functional.layer_norm(
            residual_39,
            (512,),
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_,
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_,
            1e-05,
        )
        residual_39 = (
            l_instance_modules_layers_modules_9_modules_norm_out_parameters_weight_
        ) = l_instance_modules_layers_modules_9_modules_norm_out_parameters_bias_ = None
        x_299 = torch.nn.functional.layer_norm(
            x_298,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_300 = torch._C._nn.linear(
            x_299,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_299 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_301 = torch.nn.functional.silu(x_300, inplace=False)
        x_300 = None
        x_302 = torch.nn.functional.dropout(x_301, 0.1, False, False)
        x_301 = None
        x_303 = torch._C._nn.linear(
            x_302,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_302 = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_72 = torch.nn.functional.dropout(x_303, 0.1, False, False)
        x_303 = None
        mul_21 = dropout_72 * 0.5
        dropout_72 = None
        residual_40 = x_298 + mul_21
        x_298 = mul_21 = None
        x_304 = torch.nn.functional.layer_norm(
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
            x_304,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_30 = linear_93.view(1, -1, 8, 64)
        linear_93 = None
        linear_94 = torch._C._nn.linear(
            x_304,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_20 = linear_94.view(1, -1, 8, 64)
        linear_94 = None
        linear_95 = torch._C._nn.linear(
            x_304,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_304 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_150 = p_21.transpose(-2, -1)
        p_21 = None
        matrix_bd_20 = torch.matmul(q_with_bias_v_10, transpose_150)
        q_with_bias_v_10 = transpose_150 = None
        x_305 = torch._C._nn.pad(matrix_bd_20, (1, 0), "constant", None)
        matrix_bd_20 = None
        x_306 = x_305.view(1, 8, -1, 66)
        x_305 = None
        getitem_22 = x_306[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_306 = None
        x_307 = getitem_22.view(1, 8, 66, 131)
        getitem_22 = None
        transpose_151 = k_21.transpose(-2, -1)
        k_21 = None
        matrix_ac_10 = torch.matmul(q_with_bias_u_10, transpose_151)
        q_with_bias_u_10 = transpose_151 = None
        matrix_bd_21 = x_307[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_307 = None
        add_79 = matrix_ac_10 + matrix_bd_21
        matrix_ac_10 = matrix_bd_21 = None
        scores_20 = add_79 / 8.0
        add_79 = None
        mask_10 = att_mask_4.unsqueeze(1)
        scores_21 = scores_20.masked_fill(mask_10, -10000.0)
        scores_20 = None
        softmax_10 = torch.softmax(scores_21, dim=-1)
        scores_21 = None
        attn_10 = softmax_10.masked_fill(mask_10, 0.0)
        softmax_10 = mask_10 = None
        p_attn_10 = torch.nn.functional.dropout(attn_10, 0.1, False, False)
        attn_10 = None
        x_308 = torch.matmul(p_attn_10, v_21)
        p_attn_10 = v_21 = None
        transpose_152 = x_308.transpose(1, 2)
        x_308 = None
        x_309 = transpose_152.reshape(1, -1, 512)
        transpose_152 = None
        out_10 = torch._C._nn.linear(
            x_309,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_309 = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_10_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_74 = torch.nn.functional.dropout(out_10, 0.1, False, False)
        out_10 = None
        residual_41 = residual_40 + dropout_74
        residual_40 = dropout_74 = None
        x_310 = torch.nn.functional.layer_norm(
            residual_41,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_10_modules_norm_conv_parameters_bias_
        ) = None
        x_311 = x_310.transpose(1, 2)
        x_310 = None
        x_312 = torch.conv1d(
            x_311,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_311 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_313 = torch.nn.functional.glu(x_312, dim=1)
        x_312 = None
        unsqueeze_27 = pad_mask_1.unsqueeze(1)
        x_314 = x_313.masked_fill(unsqueeze_27, 0.0)
        x_313 = unsqueeze_27 = None
        new_x_10 = torch._C._nn.pad(x_314, (8, 0), "constant", None)
        x_314 = None
        x_315 = torch.conv1d(
            new_x_10,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_10 = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_316 = x_315.transpose(1, 2)
        x_315 = None
        x_317 = torch.nn.functional.layer_norm(
            x_316,
            (512,),
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_316 = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_318 = x_317.transpose(1, 2)
        x_317 = None
        x_319 = torch.nn.functional.silu(x_318, inplace=False)
        x_318 = None
        x_320 = torch.conv1d(
            x_319,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_319 = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_321 = x_320.transpose(1, 2)
        x_320 = None
        dropout_75 = torch.nn.functional.dropout(x_321, 0.1, False, False)
        x_321 = None
        residual_42 = residual_41 + dropout_75
        residual_41 = dropout_75 = None
        x_322 = torch.nn.functional.layer_norm(
            residual_42,
            (512,),
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_323 = torch._C._nn.linear(
            x_322,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_322 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_324 = torch.nn.functional.silu(x_323, inplace=False)
        x_323 = None
        x_325 = torch.nn.functional.dropout(x_324, 0.1, False, False)
        x_324 = None
        x_326 = torch._C._nn.linear(
            x_325,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_325 = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_10_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_77 = torch.nn.functional.dropout(x_326, 0.1, False, False)
        x_326 = None
        mul_22 = dropout_77 * 0.5
        dropout_77 = None
        residual_43 = residual_42 + mul_22
        residual_42 = mul_22 = None
        x_327 = torch.nn.functional.layer_norm(
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
        x_328 = torch.nn.functional.layer_norm(
            x_327,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_329 = torch._C._nn.linear(
            x_328,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_328 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_330 = torch.nn.functional.silu(x_329, inplace=False)
        x_329 = None
        x_331 = torch.nn.functional.dropout(x_330, 0.1, False, False)
        x_330 = None
        x_332 = torch._C._nn.linear(
            x_331,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_331 = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_79 = torch.nn.functional.dropout(x_332, 0.1, False, False)
        x_332 = None
        mul_23 = dropout_79 * 0.5
        dropout_79 = None
        residual_44 = x_327 + mul_23
        x_327 = mul_23 = None
        x_333 = torch.nn.functional.layer_norm(
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
            x_333,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_33 = linear_102.view(1, -1, 8, 64)
        linear_102 = None
        linear_103 = torch._C._nn.linear(
            x_333,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_22 = linear_103.view(1, -1, 8, 64)
        linear_103 = None
        linear_104 = torch._C._nn.linear(
            x_333,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_333 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_164 = p_23.transpose(-2, -1)
        p_23 = None
        matrix_bd_22 = torch.matmul(q_with_bias_v_11, transpose_164)
        q_with_bias_v_11 = transpose_164 = None
        x_334 = torch._C._nn.pad(matrix_bd_22, (1, 0), "constant", None)
        matrix_bd_22 = None
        x_335 = x_334.view(1, 8, -1, 66)
        x_334 = None
        getitem_24 = x_335[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_335 = None
        x_336 = getitem_24.view(1, 8, 66, 131)
        getitem_24 = None
        transpose_165 = k_23.transpose(-2, -1)
        k_23 = None
        matrix_ac_11 = torch.matmul(q_with_bias_u_11, transpose_165)
        q_with_bias_u_11 = transpose_165 = None
        matrix_bd_23 = x_336[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_336 = None
        add_86 = matrix_ac_11 + matrix_bd_23
        matrix_ac_11 = matrix_bd_23 = None
        scores_22 = add_86 / 8.0
        add_86 = None
        mask_11 = att_mask_4.unsqueeze(1)
        scores_23 = scores_22.masked_fill(mask_11, -10000.0)
        scores_22 = None
        softmax_11 = torch.softmax(scores_23, dim=-1)
        scores_23 = None
        attn_11 = softmax_11.masked_fill(mask_11, 0.0)
        softmax_11 = mask_11 = None
        p_attn_11 = torch.nn.functional.dropout(attn_11, 0.1, False, False)
        attn_11 = None
        x_337 = torch.matmul(p_attn_11, v_23)
        p_attn_11 = v_23 = None
        transpose_166 = x_337.transpose(1, 2)
        x_337 = None
        x_338 = transpose_166.reshape(1, -1, 512)
        transpose_166 = None
        out_11 = torch._C._nn.linear(
            x_338,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_338 = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_11_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_81 = torch.nn.functional.dropout(out_11, 0.1, False, False)
        out_11 = None
        residual_45 = residual_44 + dropout_81
        residual_44 = dropout_81 = None
        x_339 = torch.nn.functional.layer_norm(
            residual_45,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_11_modules_norm_conv_parameters_bias_
        ) = None
        x_340 = x_339.transpose(1, 2)
        x_339 = None
        x_341 = torch.conv1d(
            x_340,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_340 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_342 = torch.nn.functional.glu(x_341, dim=1)
        x_341 = None
        unsqueeze_29 = pad_mask_1.unsqueeze(1)
        x_343 = x_342.masked_fill(unsqueeze_29, 0.0)
        x_342 = unsqueeze_29 = None
        new_x_11 = torch._C._nn.pad(x_343, (8, 0), "constant", None)
        x_343 = None
        x_344 = torch.conv1d(
            new_x_11,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_11 = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_345 = x_344.transpose(1, 2)
        x_344 = None
        x_346 = torch.nn.functional.layer_norm(
            x_345,
            (512,),
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_345 = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_347 = x_346.transpose(1, 2)
        x_346 = None
        x_348 = torch.nn.functional.silu(x_347, inplace=False)
        x_347 = None
        x_349 = torch.conv1d(
            x_348,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_348 = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_350 = x_349.transpose(1, 2)
        x_349 = None
        dropout_82 = torch.nn.functional.dropout(x_350, 0.1, False, False)
        x_350 = None
        residual_46 = residual_45 + dropout_82
        residual_45 = dropout_82 = None
        x_351 = torch.nn.functional.layer_norm(
            residual_46,
            (512,),
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_352 = torch._C._nn.linear(
            x_351,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_351 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_353 = torch.nn.functional.silu(x_352, inplace=False)
        x_352 = None
        x_354 = torch.nn.functional.dropout(x_353, 0.1, False, False)
        x_353 = None
        x_355 = torch._C._nn.linear(
            x_354,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_354 = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_11_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_84 = torch.nn.functional.dropout(x_355, 0.1, False, False)
        x_355 = None
        mul_24 = dropout_84 * 0.5
        dropout_84 = None
        residual_47 = residual_46 + mul_24
        residual_46 = mul_24 = None
        x_356 = torch.nn.functional.layer_norm(
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
        x_357 = torch.nn.functional.layer_norm(
            x_356,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_358 = torch._C._nn.linear(
            x_357,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_357 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_359 = torch.nn.functional.silu(x_358, inplace=False)
        x_358 = None
        x_360 = torch.nn.functional.dropout(x_359, 0.1, False, False)
        x_359 = None
        x_361 = torch._C._nn.linear(
            x_360,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_360 = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_86 = torch.nn.functional.dropout(x_361, 0.1, False, False)
        x_361 = None
        mul_25 = dropout_86 * 0.5
        dropout_86 = None
        residual_48 = x_356 + mul_25
        x_356 = mul_25 = None
        x_362 = torch.nn.functional.layer_norm(
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
            x_362,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_36 = linear_111.view(1, -1, 8, 64)
        linear_111 = None
        linear_112 = torch._C._nn.linear(
            x_362,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_24 = linear_112.view(1, -1, 8, 64)
        linear_112 = None
        linear_113 = torch._C._nn.linear(
            x_362,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_362 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_178 = p_25.transpose(-2, -1)
        p_25 = None
        matrix_bd_24 = torch.matmul(q_with_bias_v_12, transpose_178)
        q_with_bias_v_12 = transpose_178 = None
        x_363 = torch._C._nn.pad(matrix_bd_24, (1, 0), "constant", None)
        matrix_bd_24 = None
        x_364 = x_363.view(1, 8, -1, 66)
        x_363 = None
        getitem_26 = x_364[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_364 = None
        x_365 = getitem_26.view(1, 8, 66, 131)
        getitem_26 = None
        transpose_179 = k_25.transpose(-2, -1)
        k_25 = None
        matrix_ac_12 = torch.matmul(q_with_bias_u_12, transpose_179)
        q_with_bias_u_12 = transpose_179 = None
        matrix_bd_25 = x_365[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_365 = None
        add_93 = matrix_ac_12 + matrix_bd_25
        matrix_ac_12 = matrix_bd_25 = None
        scores_24 = add_93 / 8.0
        add_93 = None
        mask_12 = att_mask_4.unsqueeze(1)
        scores_25 = scores_24.masked_fill(mask_12, -10000.0)
        scores_24 = None
        softmax_12 = torch.softmax(scores_25, dim=-1)
        scores_25 = None
        attn_12 = softmax_12.masked_fill(mask_12, 0.0)
        softmax_12 = mask_12 = None
        p_attn_12 = torch.nn.functional.dropout(attn_12, 0.1, False, False)
        attn_12 = None
        x_366 = torch.matmul(p_attn_12, v_25)
        p_attn_12 = v_25 = None
        transpose_180 = x_366.transpose(1, 2)
        x_366 = None
        x_367 = transpose_180.reshape(1, -1, 512)
        transpose_180 = None
        out_12 = torch._C._nn.linear(
            x_367,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_367 = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_12_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_88 = torch.nn.functional.dropout(out_12, 0.1, False, False)
        out_12 = None
        residual_49 = residual_48 + dropout_88
        residual_48 = dropout_88 = None
        x_368 = torch.nn.functional.layer_norm(
            residual_49,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_12_modules_norm_conv_parameters_bias_
        ) = None
        x_369 = x_368.transpose(1, 2)
        x_368 = None
        x_370 = torch.conv1d(
            x_369,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_369 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_371 = torch.nn.functional.glu(x_370, dim=1)
        x_370 = None
        unsqueeze_31 = pad_mask_1.unsqueeze(1)
        x_372 = x_371.masked_fill(unsqueeze_31, 0.0)
        x_371 = unsqueeze_31 = None
        new_x_12 = torch._C._nn.pad(x_372, (8, 0), "constant", None)
        x_372 = None
        x_373 = torch.conv1d(
            new_x_12,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_12 = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_374 = x_373.transpose(1, 2)
        x_373 = None
        x_375 = torch.nn.functional.layer_norm(
            x_374,
            (512,),
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_374 = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_376 = x_375.transpose(1, 2)
        x_375 = None
        x_377 = torch.nn.functional.silu(x_376, inplace=False)
        x_376 = None
        x_378 = torch.conv1d(
            x_377,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_377 = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_379 = x_378.transpose(1, 2)
        x_378 = None
        dropout_89 = torch.nn.functional.dropout(x_379, 0.1, False, False)
        x_379 = None
        residual_50 = residual_49 + dropout_89
        residual_49 = dropout_89 = None
        x_380 = torch.nn.functional.layer_norm(
            residual_50,
            (512,),
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_381 = torch._C._nn.linear(
            x_380,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_380 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_382 = torch.nn.functional.silu(x_381, inplace=False)
        x_381 = None
        x_383 = torch.nn.functional.dropout(x_382, 0.1, False, False)
        x_382 = None
        x_384 = torch._C._nn.linear(
            x_383,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_383 = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_12_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_91 = torch.nn.functional.dropout(x_384, 0.1, False, False)
        x_384 = None
        mul_26 = dropout_91 * 0.5
        dropout_91 = None
        residual_51 = residual_50 + mul_26
        residual_50 = mul_26 = None
        x_385 = torch.nn.functional.layer_norm(
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
        x_386 = torch.nn.functional.layer_norm(
            x_385,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_387 = torch._C._nn.linear(
            x_386,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_386 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_388 = torch.nn.functional.silu(x_387, inplace=False)
        x_387 = None
        x_389 = torch.nn.functional.dropout(x_388, 0.1, False, False)
        x_388 = None
        x_390 = torch._C._nn.linear(
            x_389,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_389 = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_93 = torch.nn.functional.dropout(x_390, 0.1, False, False)
        x_390 = None
        mul_27 = dropout_93 * 0.5
        dropout_93 = None
        residual_52 = x_385 + mul_27
        x_385 = mul_27 = None
        x_391 = torch.nn.functional.layer_norm(
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
            x_391,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_39 = linear_120.view(1, -1, 8, 64)
        linear_120 = None
        linear_121 = torch._C._nn.linear(
            x_391,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_26 = linear_121.view(1, -1, 8, 64)
        linear_121 = None
        linear_122 = torch._C._nn.linear(
            x_391,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_391 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_192 = p_27.transpose(-2, -1)
        p_27 = None
        matrix_bd_26 = torch.matmul(q_with_bias_v_13, transpose_192)
        q_with_bias_v_13 = transpose_192 = None
        x_392 = torch._C._nn.pad(matrix_bd_26, (1, 0), "constant", None)
        matrix_bd_26 = None
        x_393 = x_392.view(1, 8, -1, 66)
        x_392 = None
        getitem_28 = x_393[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_393 = None
        x_394 = getitem_28.view(1, 8, 66, 131)
        getitem_28 = None
        transpose_193 = k_27.transpose(-2, -1)
        k_27 = None
        matrix_ac_13 = torch.matmul(q_with_bias_u_13, transpose_193)
        q_with_bias_u_13 = transpose_193 = None
        matrix_bd_27 = x_394[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_394 = None
        add_100 = matrix_ac_13 + matrix_bd_27
        matrix_ac_13 = matrix_bd_27 = None
        scores_26 = add_100 / 8.0
        add_100 = None
        mask_13 = att_mask_4.unsqueeze(1)
        scores_27 = scores_26.masked_fill(mask_13, -10000.0)
        scores_26 = None
        softmax_13 = torch.softmax(scores_27, dim=-1)
        scores_27 = None
        attn_13 = softmax_13.masked_fill(mask_13, 0.0)
        softmax_13 = mask_13 = None
        p_attn_13 = torch.nn.functional.dropout(attn_13, 0.1, False, False)
        attn_13 = None
        x_395 = torch.matmul(p_attn_13, v_27)
        p_attn_13 = v_27 = None
        transpose_194 = x_395.transpose(1, 2)
        x_395 = None
        x_396 = transpose_194.reshape(1, -1, 512)
        transpose_194 = None
        out_13 = torch._C._nn.linear(
            x_396,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_396 = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_13_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_95 = torch.nn.functional.dropout(out_13, 0.1, False, False)
        out_13 = None
        residual_53 = residual_52 + dropout_95
        residual_52 = dropout_95 = None
        x_397 = torch.nn.functional.layer_norm(
            residual_53,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_13_modules_norm_conv_parameters_bias_
        ) = None
        x_398 = x_397.transpose(1, 2)
        x_397 = None
        x_399 = torch.conv1d(
            x_398,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_398 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_400 = torch.nn.functional.glu(x_399, dim=1)
        x_399 = None
        unsqueeze_33 = pad_mask_1.unsqueeze(1)
        x_401 = x_400.masked_fill(unsqueeze_33, 0.0)
        x_400 = unsqueeze_33 = None
        new_x_13 = torch._C._nn.pad(x_401, (8, 0), "constant", None)
        x_401 = None
        x_402 = torch.conv1d(
            new_x_13,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_13 = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_403 = x_402.transpose(1, 2)
        x_402 = None
        x_404 = torch.nn.functional.layer_norm(
            x_403,
            (512,),
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_403 = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_405 = x_404.transpose(1, 2)
        x_404 = None
        x_406 = torch.nn.functional.silu(x_405, inplace=False)
        x_405 = None
        x_407 = torch.conv1d(
            x_406,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_406 = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_408 = x_407.transpose(1, 2)
        x_407 = None
        dropout_96 = torch.nn.functional.dropout(x_408, 0.1, False, False)
        x_408 = None
        residual_54 = residual_53 + dropout_96
        residual_53 = dropout_96 = None
        x_409 = torch.nn.functional.layer_norm(
            residual_54,
            (512,),
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_410 = torch._C._nn.linear(
            x_409,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_409 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_411 = torch.nn.functional.silu(x_410, inplace=False)
        x_410 = None
        x_412 = torch.nn.functional.dropout(x_411, 0.1, False, False)
        x_411 = None
        x_413 = torch._C._nn.linear(
            x_412,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_412 = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_13_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_98 = torch.nn.functional.dropout(x_413, 0.1, False, False)
        x_413 = None
        mul_28 = dropout_98 * 0.5
        dropout_98 = None
        residual_55 = residual_54 + mul_28
        residual_54 = mul_28 = None
        x_414 = torch.nn.functional.layer_norm(
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
        x_415 = torch.nn.functional.layer_norm(
            x_414,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_416 = torch._C._nn.linear(
            x_415,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_415 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_417 = torch.nn.functional.silu(x_416, inplace=False)
        x_416 = None
        x_418 = torch.nn.functional.dropout(x_417, 0.1, False, False)
        x_417 = None
        x_419 = torch._C._nn.linear(
            x_418,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_418 = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_100 = torch.nn.functional.dropout(x_419, 0.1, False, False)
        x_419 = None
        mul_29 = dropout_100 * 0.5
        dropout_100 = None
        residual_56 = x_414 + mul_29
        x_414 = mul_29 = None
        x_420 = torch.nn.functional.layer_norm(
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
            x_420,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_42 = linear_129.view(1, -1, 8, 64)
        linear_129 = None
        linear_130 = torch._C._nn.linear(
            x_420,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_28 = linear_130.view(1, -1, 8, 64)
        linear_130 = None
        linear_131 = torch._C._nn.linear(
            x_420,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_420 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_206 = p_29.transpose(-2, -1)
        p_29 = None
        matrix_bd_28 = torch.matmul(q_with_bias_v_14, transpose_206)
        q_with_bias_v_14 = transpose_206 = None
        x_421 = torch._C._nn.pad(matrix_bd_28, (1, 0), "constant", None)
        matrix_bd_28 = None
        x_422 = x_421.view(1, 8, -1, 66)
        x_421 = None
        getitem_30 = x_422[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_422 = None
        x_423 = getitem_30.view(1, 8, 66, 131)
        getitem_30 = None
        transpose_207 = k_29.transpose(-2, -1)
        k_29 = None
        matrix_ac_14 = torch.matmul(q_with_bias_u_14, transpose_207)
        q_with_bias_u_14 = transpose_207 = None
        matrix_bd_29 = x_423[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_423 = None
        add_107 = matrix_ac_14 + matrix_bd_29
        matrix_ac_14 = matrix_bd_29 = None
        scores_28 = add_107 / 8.0
        add_107 = None
        mask_14 = att_mask_4.unsqueeze(1)
        scores_29 = scores_28.masked_fill(mask_14, -10000.0)
        scores_28 = None
        softmax_14 = torch.softmax(scores_29, dim=-1)
        scores_29 = None
        attn_14 = softmax_14.masked_fill(mask_14, 0.0)
        softmax_14 = mask_14 = None
        p_attn_14 = torch.nn.functional.dropout(attn_14, 0.1, False, False)
        attn_14 = None
        x_424 = torch.matmul(p_attn_14, v_29)
        p_attn_14 = v_29 = None
        transpose_208 = x_424.transpose(1, 2)
        x_424 = None
        x_425 = transpose_208.reshape(1, -1, 512)
        transpose_208 = None
        out_14 = torch._C._nn.linear(
            x_425,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_425 = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_14_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_102 = torch.nn.functional.dropout(out_14, 0.1, False, False)
        out_14 = None
        residual_57 = residual_56 + dropout_102
        residual_56 = dropout_102 = None
        x_426 = torch.nn.functional.layer_norm(
            residual_57,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_14_modules_norm_conv_parameters_bias_
        ) = None
        x_427 = x_426.transpose(1, 2)
        x_426 = None
        x_428 = torch.conv1d(
            x_427,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_427 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_429 = torch.nn.functional.glu(x_428, dim=1)
        x_428 = None
        unsqueeze_35 = pad_mask_1.unsqueeze(1)
        x_430 = x_429.masked_fill(unsqueeze_35, 0.0)
        x_429 = unsqueeze_35 = None
        new_x_14 = torch._C._nn.pad(x_430, (8, 0), "constant", None)
        x_430 = None
        x_431 = torch.conv1d(
            new_x_14,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_14 = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_432 = x_431.transpose(1, 2)
        x_431 = None
        x_433 = torch.nn.functional.layer_norm(
            x_432,
            (512,),
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_432 = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_434 = x_433.transpose(1, 2)
        x_433 = None
        x_435 = torch.nn.functional.silu(x_434, inplace=False)
        x_434 = None
        x_436 = torch.conv1d(
            x_435,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_435 = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_437 = x_436.transpose(1, 2)
        x_436 = None
        dropout_103 = torch.nn.functional.dropout(x_437, 0.1, False, False)
        x_437 = None
        residual_58 = residual_57 + dropout_103
        residual_57 = dropout_103 = None
        x_438 = torch.nn.functional.layer_norm(
            residual_58,
            (512,),
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_439 = torch._C._nn.linear(
            x_438,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_438 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_440 = torch.nn.functional.silu(x_439, inplace=False)
        x_439 = None
        x_441 = torch.nn.functional.dropout(x_440, 0.1, False, False)
        x_440 = None
        x_442 = torch._C._nn.linear(
            x_441,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_441 = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_14_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_105 = torch.nn.functional.dropout(x_442, 0.1, False, False)
        x_442 = None
        mul_30 = dropout_105 * 0.5
        dropout_105 = None
        residual_59 = residual_58 + mul_30
        residual_58 = mul_30 = None
        x_443 = torch.nn.functional.layer_norm(
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
        x_444 = torch.nn.functional.layer_norm(
            x_443,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_445 = torch._C._nn.linear(
            x_444,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_444 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_446 = torch.nn.functional.silu(x_445, inplace=False)
        x_445 = None
        x_447 = torch.nn.functional.dropout(x_446, 0.1, False, False)
        x_446 = None
        x_448 = torch._C._nn.linear(
            x_447,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_447 = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_107 = torch.nn.functional.dropout(x_448, 0.1, False, False)
        x_448 = None
        mul_31 = dropout_107 * 0.5
        dropout_107 = None
        residual_60 = x_443 + mul_31
        x_443 = mul_31 = None
        x_449 = torch.nn.functional.layer_norm(
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
            x_449,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_45 = linear_138.view(1, -1, 8, 64)
        linear_138 = None
        linear_139 = torch._C._nn.linear(
            x_449,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_30 = linear_139.view(1, -1, 8, 64)
        linear_139 = None
        linear_140 = torch._C._nn.linear(
            x_449,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_449 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_220 = p_31.transpose(-2, -1)
        p_31 = None
        matrix_bd_30 = torch.matmul(q_with_bias_v_15, transpose_220)
        q_with_bias_v_15 = transpose_220 = None
        x_450 = torch._C._nn.pad(matrix_bd_30, (1, 0), "constant", None)
        matrix_bd_30 = None
        x_451 = x_450.view(1, 8, -1, 66)
        x_450 = None
        getitem_32 = x_451[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_451 = None
        x_452 = getitem_32.view(1, 8, 66, 131)
        getitem_32 = None
        transpose_221 = k_31.transpose(-2, -1)
        k_31 = None
        matrix_ac_15 = torch.matmul(q_with_bias_u_15, transpose_221)
        q_with_bias_u_15 = transpose_221 = None
        matrix_bd_31 = x_452[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_452 = None
        add_114 = matrix_ac_15 + matrix_bd_31
        matrix_ac_15 = matrix_bd_31 = None
        scores_30 = add_114 / 8.0
        add_114 = None
        mask_15 = att_mask_4.unsqueeze(1)
        scores_31 = scores_30.masked_fill(mask_15, -10000.0)
        scores_30 = None
        softmax_15 = torch.softmax(scores_31, dim=-1)
        scores_31 = None
        attn_15 = softmax_15.masked_fill(mask_15, 0.0)
        softmax_15 = mask_15 = None
        p_attn_15 = torch.nn.functional.dropout(attn_15, 0.1, False, False)
        attn_15 = None
        x_453 = torch.matmul(p_attn_15, v_31)
        p_attn_15 = v_31 = None
        transpose_222 = x_453.transpose(1, 2)
        x_453 = None
        x_454 = transpose_222.reshape(1, -1, 512)
        transpose_222 = None
        out_15 = torch._C._nn.linear(
            x_454,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_454 = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_15_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_109 = torch.nn.functional.dropout(out_15, 0.1, False, False)
        out_15 = None
        residual_61 = residual_60 + dropout_109
        residual_60 = dropout_109 = None
        x_455 = torch.nn.functional.layer_norm(
            residual_61,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_15_modules_norm_conv_parameters_bias_
        ) = None
        x_456 = x_455.transpose(1, 2)
        x_455 = None
        x_457 = torch.conv1d(
            x_456,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_456 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_458 = torch.nn.functional.glu(x_457, dim=1)
        x_457 = None
        unsqueeze_37 = pad_mask_1.unsqueeze(1)
        x_459 = x_458.masked_fill(unsqueeze_37, 0.0)
        x_458 = unsqueeze_37 = None
        new_x_15 = torch._C._nn.pad(x_459, (8, 0), "constant", None)
        x_459 = None
        x_460 = torch.conv1d(
            new_x_15,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_15 = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_461 = x_460.transpose(1, 2)
        x_460 = None
        x_462 = torch.nn.functional.layer_norm(
            x_461,
            (512,),
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_461 = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_463 = x_462.transpose(1, 2)
        x_462 = None
        x_464 = torch.nn.functional.silu(x_463, inplace=False)
        x_463 = None
        x_465 = torch.conv1d(
            x_464,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_464 = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_466 = x_465.transpose(1, 2)
        x_465 = None
        dropout_110 = torch.nn.functional.dropout(x_466, 0.1, False, False)
        x_466 = None
        residual_62 = residual_61 + dropout_110
        residual_61 = dropout_110 = None
        x_467 = torch.nn.functional.layer_norm(
            residual_62,
            (512,),
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_468 = torch._C._nn.linear(
            x_467,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_467 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_469 = torch.nn.functional.silu(x_468, inplace=False)
        x_468 = None
        x_470 = torch.nn.functional.dropout(x_469, 0.1, False, False)
        x_469 = None
        x_471 = torch._C._nn.linear(
            x_470,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_470 = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_15_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_112 = torch.nn.functional.dropout(x_471, 0.1, False, False)
        x_471 = None
        mul_32 = dropout_112 * 0.5
        dropout_112 = None
        residual_63 = residual_62 + mul_32
        residual_62 = mul_32 = None
        x_472 = torch.nn.functional.layer_norm(
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
        x_473 = torch.nn.functional.layer_norm(
            x_472,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_norm_feed_forward1_parameters_bias_ = (None)
        x_474 = torch._C._nn.linear(
            x_473,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_,
        )
        x_473 = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear1_parameters_bias_ = (None)
        x_475 = torch.nn.functional.silu(x_474, inplace=False)
        x_474 = None
        x_476 = torch.nn.functional.dropout(x_475, 0.1, False, False)
        x_475 = None
        x_477 = torch._C._nn.linear(
            x_476,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_,
        )
        x_476 = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward1_modules_linear2_parameters_bias_ = (None)
        dropout_114 = torch.nn.functional.dropout(x_477, 0.1, False, False)
        x_477 = None
        mul_33 = dropout_114 * 0.5
        dropout_114 = None
        residual_64 = x_472 + mul_33
        x_472 = mul_33 = None
        x_478 = torch.nn.functional.layer_norm(
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
            x_478,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_,
        )
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_q_parameters_bias_ = (None)
        q_48 = linear_147.view(1, -1, 8, 64)
        linear_147 = None
        linear_148 = torch._C._nn.linear(
            x_478,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_,
        )
        l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_k_parameters_bias_ = (None)
        k_32 = linear_148.view(1, -1, 8, 64)
        linear_148 = None
        linear_149 = torch._C._nn.linear(
            x_478,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_,
        )
        x_478 = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_v_parameters_bias_ = (None)
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
        transpose_234 = p_33.transpose(-2, -1)
        p_33 = None
        matrix_bd_32 = torch.matmul(q_with_bias_v_16, transpose_234)
        q_with_bias_v_16 = transpose_234 = None
        x_479 = torch._C._nn.pad(matrix_bd_32, (1, 0), "constant", None)
        matrix_bd_32 = None
        x_480 = x_479.view(1, 8, -1, 66)
        x_479 = None
        getitem_34 = x_480[
            (slice(None, None, None), slice(None, None, None), slice(1, None, None))
        ]
        x_480 = None
        x_481 = getitem_34.view(1, 8, 66, 131)
        getitem_34 = None
        transpose_235 = k_33.transpose(-2, -1)
        k_33 = None
        matrix_ac_16 = torch.matmul(q_with_bias_u_16, transpose_235)
        q_with_bias_u_16 = transpose_235 = None
        matrix_bd_33 = x_481[
            (
                slice(None, None, None),
                slice(None, None, None),
                slice(None, None, None),
                slice(None, 66, None),
            )
        ]
        x_481 = None
        add_121 = matrix_ac_16 + matrix_bd_33
        matrix_ac_16 = matrix_bd_33 = None
        scores_32 = add_121 / 8.0
        add_121 = None
        mask_16 = att_mask_4.unsqueeze(1)
        att_mask_4 = None
        scores_33 = scores_32.masked_fill(mask_16, -10000.0)
        scores_32 = None
        softmax_16 = torch.softmax(scores_33, dim=-1)
        scores_33 = None
        attn_16 = softmax_16.masked_fill(mask_16, 0.0)
        softmax_16 = mask_16 = None
        p_attn_16 = torch.nn.functional.dropout(attn_16, 0.1, False, False)
        attn_16 = None
        x_482 = torch.matmul(p_attn_16, v_33)
        p_attn_16 = v_33 = None
        transpose_236 = x_482.transpose(1, 2)
        x_482 = None
        x_483 = transpose_236.reshape(1, -1, 512)
        transpose_236 = None
        out_16 = torch._C._nn.linear(
            x_483,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_,
        )
        x_483 = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_weight_ = l_instance_modules_layers_modules_16_modules_self_attn_modules_linear_out_parameters_bias_ = (None)
        dropout_116 = torch.nn.functional.dropout(out_16, 0.1, False, False)
        out_16 = None
        residual_65 = residual_64 + dropout_116
        residual_64 = dropout_116 = None
        x_484 = torch.nn.functional.layer_norm(
            residual_65,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_16_modules_norm_conv_parameters_weight_ = (
            l_instance_modules_layers_modules_16_modules_norm_conv_parameters_bias_
        ) = None
        x_485 = x_484.transpose(1, 2)
        x_484 = None
        x_486 = torch.conv1d(
            x_485,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_485 = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv1_parameters_bias_ = (None)
        x_487 = torch.nn.functional.glu(x_486, dim=1)
        x_486 = None
        unsqueeze_39 = pad_mask_1.unsqueeze(1)
        pad_mask_1 = None
        x_488 = x_487.masked_fill(unsqueeze_39, 0.0)
        x_487 = unsqueeze_39 = None
        new_x_16 = torch._C._nn.pad(x_488, (8, 0), "constant", None)
        x_488 = None
        x_489 = torch.conv1d(
            new_x_16,
            l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_,
            (1,),
            (0,),
            (1,),
            512,
        )
        new_x_16 = l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_depthwise_conv_parameters_bias_ = (None)
        x_490 = x_489.transpose(1, 2)
        x_489 = None
        x_491 = torch.nn.functional.layer_norm(
            x_490,
            (512,),
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_,
            1e-05,
        )
        x_490 = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_batch_norm_parameters_bias_ = (None)
        x_492 = x_491.transpose(1, 2)
        x_491 = None
        x_493 = torch.nn.functional.silu(x_492, inplace=False)
        x_492 = None
        x_494 = torch.conv1d(
            x_493,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_,
            (1,),
            (0,),
            (1,),
            1,
        )
        x_493 = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_conv_modules_pointwise_conv2_parameters_bias_ = (None)
        x_495 = x_494.transpose(1, 2)
        x_494 = None
        dropout_117 = torch.nn.functional.dropout(x_495, 0.1, False, False)
        x_495 = None
        residual_66 = residual_65 + dropout_117
        residual_65 = dropout_117 = None
        x_496 = torch.nn.functional.layer_norm(
            residual_66,
            (512,),
            l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_,
            1e-05,
        )
        l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_norm_feed_forward2_parameters_bias_ = (None)
        x_497 = torch._C._nn.linear(
            x_496,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_,
        )
        x_496 = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear1_parameters_bias_ = (None)
        x_498 = torch.nn.functional.silu(x_497, inplace=False)
        x_497 = None
        x_499 = torch.nn.functional.dropout(x_498, 0.1, False, False)
        x_498 = None
        x_500 = torch._C._nn.linear(
            x_499,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_,
            l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_,
        )
        x_499 = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_weight_ = l_instance_modules_layers_modules_16_modules_feed_forward2_modules_linear2_parameters_bias_ = (None)
        dropout_119 = torch.nn.functional.dropout(x_500, 0.1, False, False)
        x_500 = None
        mul_34 = dropout_119 * 0.5
        dropout_119 = None
        residual_67 = residual_66 + mul_34
        residual_66 = mul_34 = None
        x_501 = torch.nn.functional.layer_norm(
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
        audio_signal_2 = torch.transpose(x_501, 1, 2)
        x_501 = None
        length_1 = length.to(dtype=torch.int64)
        length = None
        return (audio_signal_2, length_1)
