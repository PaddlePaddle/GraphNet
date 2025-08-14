import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_decode_head_modules_conv_seg_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_bias_
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_bias_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_weight_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_weight_
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_bias_ = L_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_bias_
        l_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_ = L_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_
        l_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_var_ = L_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_var_
        l_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_weight_ = L_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_weight_
        l_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_bias_ = L_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_bias_
        l_self_modules_decode_head_modules_conv_seg_parameters_weight_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_weight_
        )
        l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = (
            L_self_modules_decode_head_modules_conv_seg_parameters_bias_
        )
        conv_out = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_0_modules_conv_parameters_bias_ = (None)
        pool_out = torch.nn.functional.max_pool2d(
            l_inputs_, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        l_inputs_ = None
        pool_out_1 = torch.nn.functional.interpolate(
            pool_out, (256, 256), None, "bilinear", False
        )
        pool_out = None
        output = torch.cat([conv_out, pool_out_1], 1)
        conv_out = pool_out_1 = None
        output_1 = torch.nn.functional.batch_norm(
            output,
            l_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output = l_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_0_modules_bn_parameters_bias_ = (None)
        output_2 = torch.nn.functional.relu(output_1, inplace=False)
        output_1 = None
        conv_out_1 = torch.conv2d(
            output_2,
            l_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_1_modules_conv_parameters_bias_ = (None)
        pool_out_2 = torch.nn.functional.max_pool2d(
            output_2, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        output_2 = None
        pool_out_3 = torch.nn.functional.interpolate(
            pool_out_2, (128, 128), None, "bilinear", False
        )
        pool_out_2 = None
        output_3 = torch.cat([conv_out_1, pool_out_3], 1)
        conv_out_1 = pool_out_3 = None
        output_4 = torch.nn.functional.batch_norm(
            output_3,
            l_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_3 = l_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_1_modules_bn_parameters_bias_ = (None)
        output_5 = torch.nn.functional.relu(output_4, inplace=False)
        output_4 = None
        output_6 = torch.conv2d(
            output_5,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_7 = torch.nn.functional.relu(output_6, inplace=False)
        output_6 = None
        output_8 = torch.conv2d(
            output_7,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_7 = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_9 = torch.nn.functional.batch_norm(
            output_8,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_8 = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_10 = torch.nn.functional.relu(output_9, inplace=False)
        output_9 = None
        output_11 = torch.conv2d(
            output_10,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_10 = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_12 = torch.nn.functional.relu(output_11, inplace=False)
        output_11 = None
        output_13 = torch.conv2d(
            output_12,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_12 = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_14 = torch.nn.functional.batch_norm(
            output_13,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_13 = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_2_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_15 = torch.nn.functional.dropout(output_14, 0.1, False, False)
        output_14 = None
        add = output_15 + output_5
        output_15 = output_5 = None
        output_16 = torch.nn.functional.relu(add, inplace=False)
        add = None
        output_17 = torch.conv2d(
            output_16,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_18 = torch.nn.functional.relu(output_17, inplace=False)
        output_17 = None
        output_19 = torch.conv2d(
            output_18,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_18 = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_20 = torch.nn.functional.batch_norm(
            output_19,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_19 = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_21 = torch.nn.functional.relu(output_20, inplace=False)
        output_20 = None
        output_22 = torch.conv2d(
            output_21,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_21 = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_23 = torch.nn.functional.relu(output_22, inplace=False)
        output_22 = None
        output_24 = torch.conv2d(
            output_23,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_23 = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_25 = torch.nn.functional.batch_norm(
            output_24,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_24 = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_3_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_26 = torch.nn.functional.dropout(output_25, 0.1, False, False)
        output_25 = None
        add_1 = output_26 + output_16
        output_26 = output_16 = None
        output_27 = torch.nn.functional.relu(add_1, inplace=False)
        add_1 = None
        output_28 = torch.conv2d(
            output_27,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_29 = torch.nn.functional.relu(output_28, inplace=False)
        output_28 = None
        output_30 = torch.conv2d(
            output_29,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_29 = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_31 = torch.nn.functional.batch_norm(
            output_30,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_30 = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_32 = torch.nn.functional.relu(output_31, inplace=False)
        output_31 = None
        output_33 = torch.conv2d(
            output_32,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_32 = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_34 = torch.nn.functional.relu(output_33, inplace=False)
        output_33 = None
        output_35 = torch.conv2d(
            output_34,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_34 = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_36 = torch.nn.functional.batch_norm(
            output_35,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_35 = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_4_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_37 = torch.nn.functional.dropout(output_36, 0.1, False, False)
        output_36 = None
        add_2 = output_37 + output_27
        output_37 = output_27 = None
        output_38 = torch.nn.functional.relu(add_2, inplace=False)
        add_2 = None
        output_39 = torch.conv2d(
            output_38,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_40 = torch.nn.functional.relu(output_39, inplace=False)
        output_39 = None
        output_41 = torch.conv2d(
            output_40,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_40 = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_42 = torch.nn.functional.batch_norm(
            output_41,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_41 = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_43 = torch.nn.functional.relu(output_42, inplace=False)
        output_42 = None
        output_44 = torch.conv2d(
            output_43,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_43 = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_45 = torch.nn.functional.relu(output_44, inplace=False)
        output_44 = None
        output_46 = torch.conv2d(
            output_45,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_45 = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_47 = torch.nn.functional.batch_norm(
            output_46,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_46 = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_5_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_48 = torch.nn.functional.dropout(output_47, 0.1, False, False)
        output_47 = None
        add_3 = output_48 + output_38
        output_48 = output_38 = None
        output_49 = torch.nn.functional.relu(add_3, inplace=False)
        add_3 = None
        output_50 = torch.conv2d(
            output_49,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_51 = torch.nn.functional.relu(output_50, inplace=False)
        output_50 = None
        output_52 = torch.conv2d(
            output_51,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_51 = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_53 = torch.nn.functional.batch_norm(
            output_52,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_52 = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_54 = torch.nn.functional.relu(output_53, inplace=False)
        output_53 = None
        output_55 = torch.conv2d(
            output_54,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_54 = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_56 = torch.nn.functional.relu(output_55, inplace=False)
        output_55 = None
        output_57 = torch.conv2d(
            output_56,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_56 = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_58 = torch.nn.functional.batch_norm(
            output_57,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_57 = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_6_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_59 = torch.nn.functional.dropout(output_58, 0.1, False, False)
        output_58 = None
        add_4 = output_59 + output_49
        output_59 = output_49 = None
        output_60 = torch.nn.functional.relu(add_4, inplace=False)
        add_4 = None
        conv_out_2 = torch.conv2d(
            output_60,
            l_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_7_modules_conv_parameters_bias_ = (None)
        pool_out_4 = torch.nn.functional.max_pool2d(
            output_60, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        output_60 = None
        pool_out_5 = torch.nn.functional.interpolate(
            pool_out_4, (64, 64), None, "bilinear", False
        )
        pool_out_4 = None
        output_61 = torch.cat([conv_out_2, pool_out_5], 1)
        conv_out_2 = pool_out_5 = None
        output_62 = torch.nn.functional.batch_norm(
            output_61,
            l_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_61 = l_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_7_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_7_modules_bn_parameters_bias_ = (None)
        output_63 = torch.nn.functional.relu(output_62, inplace=False)
        output_62 = None
        output_64 = torch.conv2d(
            output_63,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_65 = torch.nn.functional.relu(output_64, inplace=False)
        output_64 = None
        output_66 = torch.conv2d(
            output_65,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_65 = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_67 = torch.nn.functional.batch_norm(
            output_66,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_66 = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_68 = torch.nn.functional.relu(output_67, inplace=False)
        output_67 = None
        output_69 = torch.conv2d(
            output_68,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (2, 0),
            (2, 1),
            1,
        )
        output_68 = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_70 = torch.nn.functional.relu(output_69, inplace=False)
        output_69 = None
        output_71 = torch.conv2d(
            output_70,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 2),
            (1, 2),
            1,
        )
        output_70 = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_72 = torch.nn.functional.batch_norm(
            output_71,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_71 = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_8_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_73 = torch.nn.functional.dropout(output_72, 0.1, False, False)
        output_72 = None
        add_5 = output_73 + output_63
        output_73 = output_63 = None
        output_74 = torch.nn.functional.relu(add_5, inplace=False)
        add_5 = None
        output_75 = torch.conv2d(
            output_74,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_76 = torch.nn.functional.relu(output_75, inplace=False)
        output_75 = None
        output_77 = torch.conv2d(
            output_76,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_76 = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_78 = torch.nn.functional.batch_norm(
            output_77,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_77 = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_79 = torch.nn.functional.relu(output_78, inplace=False)
        output_78 = None
        output_80 = torch.conv2d(
            output_79,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (4, 0),
            (4, 1),
            1,
        )
        output_79 = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_81 = torch.nn.functional.relu(output_80, inplace=False)
        output_80 = None
        output_82 = torch.conv2d(
            output_81,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 4),
            (1, 4),
            1,
        )
        output_81 = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_83 = torch.nn.functional.batch_norm(
            output_82,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_82 = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_9_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_84 = torch.nn.functional.dropout(output_83, 0.1, False, False)
        output_83 = None
        add_6 = output_84 + output_74
        output_84 = output_74 = None
        output_85 = torch.nn.functional.relu(add_6, inplace=False)
        add_6 = None
        output_86 = torch.conv2d(
            output_85,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_87 = torch.nn.functional.relu(output_86, inplace=False)
        output_86 = None
        output_88 = torch.conv2d(
            output_87,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_87 = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_89 = torch.nn.functional.batch_norm(
            output_88,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_88 = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_90 = torch.nn.functional.relu(output_89, inplace=False)
        output_89 = None
        output_91 = torch.conv2d(
            output_90,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (8, 0),
            (8, 1),
            1,
        )
        output_90 = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_92 = torch.nn.functional.relu(output_91, inplace=False)
        output_91 = None
        output_93 = torch.conv2d(
            output_92,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 8),
            (1, 8),
            1,
        )
        output_92 = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_94 = torch.nn.functional.batch_norm(
            output_93,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_93 = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_10_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_95 = torch.nn.functional.dropout(output_94, 0.1, False, False)
        output_94 = None
        add_7 = output_95 + output_85
        output_95 = output_85 = None
        output_96 = torch.nn.functional.relu(add_7, inplace=False)
        add_7 = None
        output_97 = torch.conv2d(
            output_96,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_98 = torch.nn.functional.relu(output_97, inplace=False)
        output_97 = None
        output_99 = torch.conv2d(
            output_98,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_98 = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_100 = torch.nn.functional.batch_norm(
            output_99,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_99 = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_101 = torch.nn.functional.relu(output_100, inplace=False)
        output_100 = None
        output_102 = torch.conv2d(
            output_101,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (16, 0),
            (16, 1),
            1,
        )
        output_101 = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_103 = torch.nn.functional.relu(output_102, inplace=False)
        output_102 = None
        output_104 = torch.conv2d(
            output_103,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 16),
            (1, 16),
            1,
        )
        output_103 = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_105 = torch.nn.functional.batch_norm(
            output_104,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_104 = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_11_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_106 = torch.nn.functional.dropout(output_105, 0.1, False, False)
        output_105 = None
        add_8 = output_106 + output_96
        output_106 = output_96 = None
        output_107 = torch.nn.functional.relu(add_8, inplace=False)
        add_8 = None
        output_108 = torch.conv2d(
            output_107,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_109 = torch.nn.functional.relu(output_108, inplace=False)
        output_108 = None
        output_110 = torch.conv2d(
            output_109,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_109 = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_111 = torch.nn.functional.batch_norm(
            output_110,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_110 = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_112 = torch.nn.functional.relu(output_111, inplace=False)
        output_111 = None
        output_113 = torch.conv2d(
            output_112,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (2, 0),
            (2, 1),
            1,
        )
        output_112 = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_114 = torch.nn.functional.relu(output_113, inplace=False)
        output_113 = None
        output_115 = torch.conv2d(
            output_114,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 2),
            (1, 2),
            1,
        )
        output_114 = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_116 = torch.nn.functional.batch_norm(
            output_115,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_115 = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_12_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_117 = torch.nn.functional.dropout(output_116, 0.1, False, False)
        output_116 = None
        add_9 = output_117 + output_107
        output_117 = output_107 = None
        output_118 = torch.nn.functional.relu(add_9, inplace=False)
        add_9 = None
        output_119 = torch.conv2d(
            output_118,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_120 = torch.nn.functional.relu(output_119, inplace=False)
        output_119 = None
        output_121 = torch.conv2d(
            output_120,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_120 = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_122 = torch.nn.functional.batch_norm(
            output_121,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_121 = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_123 = torch.nn.functional.relu(output_122, inplace=False)
        output_122 = None
        output_124 = torch.conv2d(
            output_123,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (4, 0),
            (4, 1),
            1,
        )
        output_123 = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_125 = torch.nn.functional.relu(output_124, inplace=False)
        output_124 = None
        output_126 = torch.conv2d(
            output_125,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 4),
            (1, 4),
            1,
        )
        output_125 = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_127 = torch.nn.functional.batch_norm(
            output_126,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_126 = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_13_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_128 = torch.nn.functional.dropout(output_127, 0.1, False, False)
        output_127 = None
        add_10 = output_128 + output_118
        output_128 = output_118 = None
        output_129 = torch.nn.functional.relu(add_10, inplace=False)
        add_10 = None
        output_130 = torch.conv2d(
            output_129,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_131 = torch.nn.functional.relu(output_130, inplace=False)
        output_130 = None
        output_132 = torch.conv2d(
            output_131,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_131 = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_133 = torch.nn.functional.batch_norm(
            output_132,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_132 = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_134 = torch.nn.functional.relu(output_133, inplace=False)
        output_133 = None
        output_135 = torch.conv2d(
            output_134,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (8, 0),
            (8, 1),
            1,
        )
        output_134 = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_136 = torch.nn.functional.relu(output_135, inplace=False)
        output_135 = None
        output_137 = torch.conv2d(
            output_136,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 8),
            (1, 8),
            1,
        )
        output_136 = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_138 = torch.nn.functional.batch_norm(
            output_137,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_137 = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_14_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_139 = torch.nn.functional.dropout(output_138, 0.1, False, False)
        output_138 = None
        add_11 = output_139 + output_129
        output_139 = output_129 = None
        output_140 = torch.nn.functional.relu(add_11, inplace=False)
        add_11 = None
        output_141 = torch.conv2d(
            output_140,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_142 = torch.nn.functional.relu(output_141, inplace=False)
        output_141 = None
        output_143 = torch.conv2d(
            output_142,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_142 = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_144 = torch.nn.functional.batch_norm(
            output_143,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_143 = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_145 = torch.nn.functional.relu(output_144, inplace=False)
        output_144 = None
        output_146 = torch.conv2d(
            output_145,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (16, 0),
            (16, 1),
            1,
        )
        output_145 = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_147 = torch.nn.functional.relu(output_146, inplace=False)
        output_146 = None
        output_148 = torch.conv2d(
            output_147,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 16),
            (1, 16),
            1,
        )
        output_147 = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_149 = torch.nn.functional.batch_norm(
            output_148,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_148 = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_encoder_modules_15_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_150 = torch.nn.functional.dropout(output_149, 0.1, False, False)
        output_149 = None
        add_12 = output_150 + output_140
        output_150 = output_140 = None
        output_151 = torch.nn.functional.relu(add_12, inplace=False)
        add_12 = None
        output_152 = torch.conv_transpose2d(
            output_151,
            l_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
            (1, 1),
        )
        output_151 = l_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_0_modules_conv_parameters_bias_ = (None)
        output_153 = torch.nn.functional.batch_norm(
            output_152,
            l_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_152 = l_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_0_modules_bn_parameters_bias_ = (None)
        output_154 = torch.nn.functional.relu(output_153, inplace=False)
        output_153 = None
        output_155 = torch.conv2d(
            output_154,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_156 = torch.nn.functional.relu(output_155, inplace=False)
        output_155 = None
        output_157 = torch.conv2d(
            output_156,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_156 = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_158 = torch.nn.functional.batch_norm(
            output_157,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_157 = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_159 = torch.nn.functional.relu(output_158, inplace=False)
        output_158 = None
        output_160 = torch.conv2d(
            output_159,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_159 = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_161 = torch.nn.functional.relu(output_160, inplace=False)
        output_160 = None
        output_162 = torch.conv2d(
            output_161,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_161 = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_163 = torch.nn.functional.batch_norm(
            output_162,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_162 = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_1_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_164 = torch.nn.functional.dropout(output_163, 0, False, False)
        output_163 = None
        add_13 = output_164 + output_154
        output_164 = output_154 = None
        output_165 = torch.nn.functional.relu(add_13, inplace=False)
        add_13 = None
        output_166 = torch.conv2d(
            output_165,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_167 = torch.nn.functional.relu(output_166, inplace=False)
        output_166 = None
        output_168 = torch.conv2d(
            output_167,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_167 = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_169 = torch.nn.functional.batch_norm(
            output_168,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_168 = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_170 = torch.nn.functional.relu(output_169, inplace=False)
        output_169 = None
        output_171 = torch.conv2d(
            output_170,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_170 = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_172 = torch.nn.functional.relu(output_171, inplace=False)
        output_171 = None
        output_173 = torch.conv2d(
            output_172,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_172 = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_174 = torch.nn.functional.batch_norm(
            output_173,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_173 = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_2_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_175 = torch.nn.functional.dropout(output_174, 0, False, False)
        output_174 = None
        add_14 = output_175 + output_165
        output_175 = output_165 = None
        output_176 = torch.nn.functional.relu(add_14, inplace=False)
        add_14 = None
        output_177 = torch.conv_transpose2d(
            output_176,
            l_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
            (1, 1),
        )
        output_176 = l_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_3_modules_conv_parameters_bias_ = (None)
        output_178 = torch.nn.functional.batch_norm(
            output_177,
            l_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_177 = l_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_3_modules_bn_parameters_bias_ = (None)
        output_179 = torch.nn.functional.relu(output_178, inplace=False)
        output_178 = None
        output_180 = torch.conv2d(
            output_179,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_181 = torch.nn.functional.relu(output_180, inplace=False)
        output_180 = None
        output_182 = torch.conv2d(
            output_181,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_181 = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_183 = torch.nn.functional.batch_norm(
            output_182,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_182 = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_184 = torch.nn.functional.relu(output_183, inplace=False)
        output_183 = None
        output_185 = torch.conv2d(
            output_184,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_184 = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_186 = torch.nn.functional.relu(output_185, inplace=False)
        output_185 = None
        output_187 = torch.conv2d(
            output_186,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_186 = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_188 = torch.nn.functional.batch_norm(
            output_187,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_187 = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_4_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_189 = torch.nn.functional.dropout(output_188, 0, False, False)
        output_188 = None
        add_15 = output_189 + output_179
        output_189 = output_179 = None
        output_190 = torch.nn.functional.relu(add_15, inplace=False)
        add_15 = None
        output_191 = torch.conv2d(
            output_190,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_0_parameters_bias_ = (None)
        output_192 = torch.nn.functional.relu(output_191, inplace=False)
        output_191 = None
        output_193 = torch.conv2d(
            output_192,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_192 = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_2_parameters_bias_ = (None)
        output_194 = torch.nn.functional.batch_norm(
            output_193,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_193 = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_3_parameters_bias_ = (None)
        output_195 = torch.nn.functional.relu(output_194, inplace=False)
        output_194 = None
        output_196 = torch.conv2d(
            output_195,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_bias_,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        output_195 = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_5_parameters_bias_ = (None)
        output_197 = torch.nn.functional.relu(output_196, inplace=False)
        output_196 = None
        output_198 = torch.conv2d(
            output_197,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_bias_,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        output_197 = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_7_parameters_bias_ = (None)
        output_199 = torch.nn.functional.batch_norm(
            output_198,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_weight_,
            l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        output_198 = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_mean_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_buffers_running_var_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_weight_ = l_self_modules_backbone_modules_decoder_modules_5_modules_convs_layers_modules_8_parameters_bias_ = (None)
        output_200 = torch.nn.functional.dropout(output_199, 0, False, False)
        output_199 = None
        add_16 = output_200 + output_190
        output_200 = output_190 = None
        output_201 = torch.nn.functional.relu(add_16, inplace=False)
        add_16 = None
        x = torch.conv2d(
            output_201,
            l_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        output_201 = l_self_modules_decode_head_modules_convs_modules_0_modules_conv_parameters_weight_ = (None)
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_weight_,
            l_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = l_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_decode_head_modules_convs_modules_0_modules_bn_buffers_running_var_ = l_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_weight_ = l_self_modules_decode_head_modules_convs_modules_0_modules_bn_parameters_bias_ = (None)
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        feat = torch.nn.functional.dropout2d(x_2, 0.1, False, False)
        x_2 = None
        output_202 = torch.conv2d(
            feat,
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_,
            l_self_modules_decode_head_modules_conv_seg_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        feat = (
            l_self_modules_decode_head_modules_conv_seg_parameters_weight_
        ) = l_self_modules_decode_head_modules_conv_seg_parameters_bias_ = None
        return (output_202,)
