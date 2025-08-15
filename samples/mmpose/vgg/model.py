import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_inputs_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_: torch.Tensor,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_final_layer_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_inputs_ = L_inputs_
        l_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_bias_
        l_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_weight_ = L_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_weight_
        l_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_bias_ = L_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_bias_
        l_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_mean_ = L_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_mean_
        l_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_var_ = L_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_var_
        l_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_weight_ = L_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_weight_
        l_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_bias_ = L_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_bias_
        l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_
        )
        l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        )
        l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = (
            L_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_
        )
        l_self_modules_head_modules_final_layer_parameters_weight_ = (
            L_self_modules_head_modules_final_layer_parameters_weight_
        )
        l_self_modules_head_modules_final_layer_parameters_bias_ = (
            L_self_modules_head_modules_final_layer_parameters_bias_
        )
        x = torch.conv2d(
            l_inputs_,
            l_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_inputs_ = l_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_0_modules_conv_parameters_bias_ = (None)
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = l_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_0_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_0_modules_bn_parameters_bias_ = (None)
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_1_modules_conv_parameters_bias_ = (None)
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = l_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_1_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_1_modules_bn_parameters_bias_ = (None)
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.nn.functional.max_pool2d(
            x_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_6 = l_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_3_modules_conv_parameters_bias_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_3_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_3_modules_bn_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_4_modules_conv_parameters_bias_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_4_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_4_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_13 = torch.nn.functional.max_pool2d(
            x_12, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_6_modules_conv_parameters_bias_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_6_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_6_modules_bn_parameters_bias_ = (None)
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_7_modules_conv_parameters_bias_ = (None)
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_17 = l_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_7_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_7_modules_bn_parameters_bias_ = (None)
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_8_modules_conv_parameters_bias_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_8_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_8_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.nn.functional.max_pool2d(
            x_22, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_22 = None
        x_24 = torch.conv2d(
            x_23,
            l_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_23 = l_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_10_modules_conv_parameters_bias_ = (None)
        x_25 = torch.nn.functional.batch_norm(
            x_24,
            l_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_24 = l_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_10_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_10_modules_bn_parameters_bias_ = (None)
        x_26 = torch.nn.functional.relu(x_25, inplace=True)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_11_modules_conv_parameters_bias_ = (None)
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = l_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_11_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_11_modules_bn_parameters_bias_ = (None)
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_29 = l_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_12_modules_conv_parameters_bias_ = (None)
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = l_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_12_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_12_modules_bn_parameters_bias_ = (None)
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_33 = torch.nn.functional.max_pool2d(
            x_32, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_32 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_33 = l_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_14_modules_conv_parameters_bias_ = (None)
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = l_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_14_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_14_modules_bn_parameters_bias_ = (None)
        x_36 = torch.nn.functional.relu(x_35, inplace=True)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_36 = l_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_15_modules_conv_parameters_bias_ = (None)
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = l_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_15_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_15_modules_bn_parameters_bias_ = (None)
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_weight_ = l_self_modules_backbone_modules_features_modules_16_modules_conv_parameters_bias_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_mean_,
            l_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_var_,
            l_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_weight_,
            l_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_mean_ = l_self_modules_backbone_modules_features_modules_16_modules_bn_buffers_running_var_ = l_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_weight_ = l_self_modules_backbone_modules_features_modules_16_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.nn.functional.max_pool2d(
            x_42, 2, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_42 = None
        input_1 = torch.conv_transpose2d(
            x_43,
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        x_43 = (
            l_self_modules_head_modules_deconv_layers_modules_0_parameters_weight_
        ) = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_1_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv_transpose2d(
            input_3,
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_3 = (
            l_self_modules_head_modules_deconv_layers_modules_3_parameters_weight_
        ) = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_4_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv_transpose2d(
            input_6,
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (0, 0),
            1,
            (1, 1),
        )
        input_6 = (
            l_self_modules_head_modules_deconv_layers_modules_6_parameters_weight_
        ) = None
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_,
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_,
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_mean_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_buffers_running_var_
        ) = (
            l_self_modules_head_modules_deconv_layers_modules_7_parameters_weight_
        ) = l_self_modules_head_modules_deconv_layers_modules_7_parameters_bias_ = None
        input_9 = torch.nn.functional.relu(input_8, inplace=True)
        input_8 = None
        x_44 = torch.conv2d(
            input_9,
            l_self_modules_head_modules_final_layer_parameters_weight_,
            l_self_modules_head_modules_final_layer_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_9 = (
            l_self_modules_head_modules_final_layer_parameters_weight_
        ) = l_self_modules_head_modules_final_layer_parameters_bias_ = None
        return (x_44,)
